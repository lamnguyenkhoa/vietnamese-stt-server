"""Create a self-signed HTTPS certificate for the server and enable it in config.ini.

Browsers only allow microphone access on https:// pages (or http://localhost), so the
web page can't record when the server is reached over plain http by IP. This writes
cert.pem/key.pem to the app folder, valid for localhost, this machine's hostname, its
IPv4 addresses, and any extra hostnames/IPs given on the command line:

    python make_cert.py                      # auto-detected names/IPs only
    python make_cert.py 203.0.113.5 stt.lan  # plus these

Browsers will warn once that the certificate is self-signed; accept it to continue.
Re-run after the machine's IP changes.
"""
import datetime
import ipaddress
import re
import socket
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import paths

CERT_FILE = paths.APP_DIR / "cert.pem"
KEY_FILE = paths.APP_DIR / "key.pem"
CONFIG_FILE = paths.APP_DIR / "config.ini"


def local_ipv4s() -> "set[str]":
    ips = {"127.0.0.1"}
    try:
        ips.update(info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except socket.gaierror:
        pass
    # The address of the interface that would route outward (no packet is sent).
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return ips


def build_san(extra: "list[str]") -> "tuple[list[str], list[str]]":
    dns_names = {"localhost", socket.gethostname()}
    ips = local_ipv4s()
    for name in extra:
        try:
            ips.add(str(ipaddress.ip_address(name)))
        except ValueError:
            dns_names.add(name)
    return sorted(dns_names), sorted(ips, key=ipaddress.ip_address)


def enable_in_config() -> bool:
    """Point SSL_CERTFILE/SSL_KEYFILE in config.ini at the new files. Returns False if
    there is no config.ini (dev checkout) or it lacks those keys."""
    if not CONFIG_FILE.is_file():
        return False
    text = CONFIG_FILE.read_text(encoding="utf-8")
    new_text = re.sub(r"(?m)^SSL_CERTFILE=.*$", f"SSL_CERTFILE={CERT_FILE.name}", text)
    new_text = re.sub(r"(?m)^SSL_KEYFILE=.*$", f"SSL_KEYFILE={KEY_FILE.name}", new_text)
    if "SSL_CERTFILE=" not in new_text or "SSL_KEYFILE=" not in new_text:
        return False
    CONFIG_FILE.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    dns_names, ips = build_san(sys.argv[1:])

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Vietnamese STT Server")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(n) for n in dns_names]
                + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    KEY_FILE.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Wrote {CERT_FILE} and {KEY_FILE}, valid for:")
    for entry in dns_names + ips:
        print(f"  {entry}")
    if enable_in_config():
        print(f"Enabled HTTPS in {CONFIG_FILE.name}; restart the server to apply.")
    else:
        print(f"Start the server with: --ssl-certfile {CERT_FILE.name} --ssl-keyfile {KEY_FILE.name}")
    print("Then open https://<one of the addresses above>:<port>/static/index.html")


if __name__ == "__main__":
    main()
