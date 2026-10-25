# Phantom Privacy

A small, personal **privacy and self-audit toolkit** written in plain Python
(standard library only — easy to read while you're learning).

It's built around one idea: **your phone number is a key to your identity**, so
stop handing out your real one, understand what you expose, and reduce it.

## What it does

| Command | What it does |
| --- | --- |
| `phone` | Guidance on protecting your phone number: why it matters, VoIP/masking numbers (MySudo and alternatives), SS7 awareness, and a checklist. |
| `footprint` | An interactive questionnaire that scores your privacy posture and gives prioritised recommendations. |
| `scan` | A **defensive** nmap wrapper that audits *your own* network to show what services you expose. |

## Requirements

- Python 3.10+
- `nmap` installed on your system (only needed for the `scan` command):
  - Debian/Ubuntu: `sudo apt install nmap`
  - macOS: `brew install nmap`

No third-party Python packages are required. `pytest` is only needed to run the tests.

## Usage

Run it as a module from inside the `phantom-privacy/` directory:

```bash
python -m phantom_privacy phone
python -m phantom_privacy footprint
python -m phantom_privacy scan                 # scans 127.0.0.1 by default
python -m phantom_privacy scan 192.168.1.0/24  # your own LAN
python -m phantom_privacy scan 192.168.1.10 --ports 1-1024
```

## Safety / scope of the scanner

The `scan` command is for **self-audit only**:

- By default it only accepts **loopback and private (RFC1918)** addresses —
  i.e. your own machine and your own LAN.
- Scanning any public address requires the explicit `--allow-public` flag and is
  intended **only for hosts you own or are authorised to test**.
- It uses nmap's plain, non-evasive defaults (a basic TCP connect scan). There
  are no stealth or evasion options.

Only scan networks and devices you own or have explicit permission to test.

## Running the tests

```bash
cd phantom-privacy
python -m pytest
```

## Project layout

```
phantom-privacy/
├── README.md
├── phantom_privacy/
│   ├── __init__.py
│   ├── __main__.py        # enables `python -m phantom_privacy`
│   ├── cli.py             # argparse CLI wiring the modules together
│   ├── phone_privacy.py   # phone-number guidance (text only)
│   ├── footprint.py       # footprint questionnaire + scoring
│   └── network_scan.py    # defensive nmap wrapper
└── tests/
    ├── test_footprint.py
    └── test_network_scan.py
```
