"""SshEndpoint.parse handles the shapes Prime might return."""
from __future__ import annotations

from pathlib import Path

import pytest

from primejob.backend.ssh import SshEndpoint


def test_parse_ssh_command_string() -> None:
    e = SshEndpoint.parse("ssh root@1.2.3.4 -p 12345")
    assert e.host == "1.2.3.4"
    assert e.port == 12345
    assert e.user == "root"


def test_parse_userhost_colon_port() -> None:
    e = SshEndpoint.parse("user1@host.example.com:2222")
    assert e.host == "host.example.com"
    assert e.port == 2222
    assert e.user == "user1"


def test_parse_default_port() -> None:
    e = SshEndpoint.parse("root@5.6.7.8")
    assert e.port == 22


def test_parse_list_form() -> None:
    e = SshEndpoint.parse(["ubuntu@204.12.168.84"])
    assert e.host == "204.12.168.84"
    assert e.port == 22
    assert e.user == "ubuntu"


def test_parse_dict() -> None:
    e = SshEndpoint.parse({"host": "h", "port": "9000", "user": "ubuntu"})
    assert e.host == "h"
    assert e.port == 9000
    assert e.user == "ubuntu"


def test_parse_dict_ip_alias() -> None:
    e = SshEndpoint.parse({"ip": "10.0.0.1", "port": 22, "username": "dev"})
    assert e.host == "10.0.0.1"
    assert e.user == "dev"


def test_parse_none_raises() -> None:
    with pytest.raises(ValueError):
        SshEndpoint.parse(None)


def test_parse_unparseable_raises() -> None:
    with pytest.raises(ValueError):
        SshEndpoint.parse("totally not an ssh thing")


def test_parse_with_key_path() -> None:
    key = Path("/tmp/test_key")
    e = SshEndpoint.parse("root@1.2.3.4", key_path=key)
    assert e.key_path == key
