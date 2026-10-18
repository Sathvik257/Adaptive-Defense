# Isolated traffic collection

This lab sends bounded HTTP browsing, TCP connection probes (21 local ports),
and thirty rejected login requests to a disposable server. It does not execute
exploits, guess real credentials, or target any external machine. Failed logins
are a scenario label, not proof of a successful brute-force attack.

Prerequisite: Docker Desktop with its Linux engine running. Build images before
starting the isolated runtime network:

```powershell
docker compose -f lab/compose.yaml build
```

Run the following collection sequence; wait for tcpdump to flush before stopping:

```powershell
docker compose -f lab/compose.yaml up -d
docker compose -f lab/compose.yaml wait collector
docker compose -f lab/compose.yaml logs generator
docker compose -f lab/compose.yaml down
```

Preserve capture/traffic.pcap and events.json before another run; reruns overwrite
these names. The collector records the generator network namespace. There are no
published ports, Docker socket mounts or host networking. The sole runtime
network is internal; image builds require Internet package access.

Status: configuration prepared; execution blocked by the local Docker engine
being unavailable. No PCAP or external validation result has been fabricated.

## Evaluation contract

Extract bidirectional flows using the SAME exporter/version/options that produced
the training dataset. Associate flows with scenario windows from events.json;
discard ambiguous boundary flows and document packet loss. Keep label and family
outside the model feature matrix. Do not rename incompatible exporter fields or
fill missing UNSW features with zeros merely to make prediction succeed.

The bundled synthetic model is not a validated PCAP model. Its features include
dataset-specific state/service fields, and its distributions are artificial.
Use `adaptive-predict --run <trusted-run> --csv <compatible-flows.csv> --output
<new-results.csv> --mode frozen` only after verifying units and semantics. Freeze
the model before collection, report false alarms and per-scenario detection with
counts, and retain packet capture hashes. Collect several independent sessions;
do not split adjacent packets from one session into training and testing.
