# Transcribing with live preview

The server offers two ways to transcribe, designed to be used **together**:

| Endpoint | What it gives you | Accuracy | When |
|-|-|-|-|
| `WS /ws/stream` | Rough text that grows while the user speaks | Lower | During recording |
| `POST /transcribe` | Final text for the whole recording | Best | After recording stops |

The preview is for responsiveness: words appear as the user talks. Its accuracy is
lower because it only ever sees short windows of audio. The final result from
`/transcribe` sees the whole recording, so it should **always replace** the preview.

```
user speaks ──► mic ──┬──► PCM chunks ──► WS /ws/stream ──► {"text": "rough…"}  (show, grey)
                      │
                      └──► recorded file ──(on stop)──► POST /transcribe ──► {"text": "final"}  (replace)
```

If you don't need the preview, skip the WebSocket and just POST the recording.

## Using the test page

1. Start the server over HTTPS (see [the README](../README.md#run-the-server)).
   Browsers only allow microphone access on `https://` or `localhost`.
2. Open `https://<server-ip>:8123/static/index.html` and accept the certificate warning.
3. Leave **Live preview** checked and click **Start**. Allow the microphone.
4. Speak. Grey text is the live preview.
5. Click **Stop**, or stay silent for about 2 seconds. The grey text is replaced by
   the final result in normal black text.

An empty recording shows `(no speech detected)`. The API itself returns `""` in that case.

## Building your own client

### Step 1: open the preview socket when recording starts

Connect to `ws://<host>:<port>/ws/stream`, or `wss://` when the server uses HTTPS.

**Send** binary frames of raw audio:

- 16-bit signed integer PCM, little-endian (PCM16LE)
- mono
- 16,000 Hz
- no header (not a WAV file), any chunk size (~50–250 ms per frame works well)

**Receive** JSON text messages, roughly once per second while there's new audio:

```json
{"text": "tìm người mặc áo đỏ"}
```

Each message is the **whole preview so far**, not just new words. Replace what
you display; don't append. During silence the text may stay the same or be `""`.

### Step 2: record the full audio at the same time

Keep your own copy of the recording, in any format ffmpeg can decode (WAV, MP3,
WebM/Opus from a browser's `MediaRecorder`, and so on). Do not rebuild it from the
preview stream. Use the original recording, ideally at the device's native quality.

### Step 3: when recording stops

1. **Close the WebSocket.** The server stops preview work for this session.
2. **Ignore any preview messages that arrive after this point.** One may still
   be in flight and must not overwrite the final text.
3. **POST the recording** to `/transcribe` as multipart form data with the field
   name `file`:

   ```
   POST /transcribe
   Content-Type: multipart/form-data
   file=<recording>
   ```

   Response:

   ```json
   {"text": "tìm người mặc áo đỏ quần đen."}
   ```

4. **Replace the preview** with this text. `""` means no speech was detected.

A short recording usually returns in well under a second on a GPU. Expect longer on
CPU, especially with the `medium` model.

## Example: browser (JavaScript)

The full working version, with auto-stop on silence, is
[static/index.html](../static/index.html). This example shows only the essentials.

```js
let ws, recorder, chunks, audioCtx, processor, finished;

async function start(onPreview) {
  finished = false;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  // 1. Live preview socket
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/stream`);
  ws.onmessage = (e) => { if (!finished) onPreview(JSON.parse(e.data).text); };

  // 2. Full recording for the final pass
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.start();

  // Feed the preview: browser audio is Float32 at 44.1/48 kHz -> PCM16 at 16 kHz
  audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(stream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (e) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const ratio = audioCtx.sampleRate / 16000;
    const pcm = new Int16Array(Math.floor(input.length / ratio));
    for (let i = 0; i < pcm.length; i++) {
      const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
      pcm[i] = s * 0x7fff;
    }
    ws.send(pcm.buffer);
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);
}

function stop() {
  // 3. Stop the preview first, and ignore late preview messages
  finished = true;
  ws.close();
  processor.disconnect();
  audioCtx.close();

  return new Promise((resolve) => {
    recorder.onstop = async () => {
      recorder.stream.getTracks().forEach((t) => t.stop());
      const form = new FormData();
      form.append("file", new Blob(chunks, { type: recorder.mimeType }), "recording.webm");
      const res = await fetch("/transcribe", { method: "POST", body: form });
      resolve((await res.json()).text); // 4. Final text: replaces the preview
    };
    recorder.stop();
  });
}

// Usage
// await start((text) => (box.textContent = text));   // grey preview
// box.textContent = await stop();                    // final text
```

## Example: Python

This example streams a file instead of a microphone. The protocol is the same for a
microphone: send PCM chunks as they're captured, then POST the full audio.

```python
import asyncio, json, ssl, subprocess
import httpx, websockets

WS_URL = "ws://127.0.0.1:8123/ws/stream"   # wss:// if the server uses HTTPS
HTTP_URL = "http://127.0.0.1:8123/transcribe"  # https:// if the server uses HTTPS
AUDIO = "recording.mp3"
# Server on HTTPS with a self-signed certificate (make_cert)? Point this at its cert.pem.
CAFILE = None
ssl_ctx = ssl.create_default_context(cafile=CAFILE) if CAFILE else None

async def main():
    # Decode to raw PCM16LE mono 16 kHz, the format /ws/stream expects.
    pcm = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", AUDIO, "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout

    async with websockets.connect(WS_URL, ssl=ssl_ctx) as ws:
        async def show_preview():
            async for msg in ws:
                print("preview:", json.loads(msg)["text"])
        preview = asyncio.create_task(show_preview())

        chunk = 3200                                # 100 ms of audio
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i + chunk])
            await asyncio.sleep(0.1)                # simulate real time
        preview.cancel()                            # stop listening, then close

    async with httpx.AsyncClient(timeout=300, verify=ssl_ctx or True) as client:
        with open(AUDIO, "rb") as f:
            res = await client.post(HTTP_URL, files={"file": f})
    print("final:", res.json()["text"])

asyncio.run(main())
```

## Tuning and behavior

Set these as environment variables. In a portable build, add them as new lines in
`config.ini`; `run.bat`/`run.sh` passes every line on to the server.

| Setting | Default | Effect |
|-|-|-|
| `STREAM_UPDATE_SECONDS` | `1.0` | The preview reruns once at least this much new audio has arrived. Lower = more frequent updates, more GPU/CPU load. |
| `STREAM_WINDOW_SECONDS` | `8.0` | The preview transcribes at most this much audio per pass, then locks that text in and starts a new window. Longer = better preview accuracy, slower passes. |

- **Silence returns `""`.** Both endpoints run voice activity detection first.
  Silence or background noise produces an empty result, not invented text.
- **Slow machines don't fall behind.** Each socket runs at most one preview pass at
  a time. Audio that arrives during a pass is included in the next one, so updates
  just come less often.
- **The preview and the final pass share the model.** A preview pass still running
  at stop time can delay the final result by up to one pass. This is why you close
  the socket before POSTing.
- **Preview text can change.** The current window is re-transcribed as more audio
  arrives, so the last few words may be revised.
