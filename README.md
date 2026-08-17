# Frigate Pro

Git-managed deployment files for the Frigate NVR running on the media server.

## Layout

- `compose.yaml` runs the official Frigate image with the server's current ports, Intel GPU mapping, storage mounts, and 1 GB shared-memory allocation.
- The live Frigate configuration remains at `/DATA/AppData/frigate/config/config.yaml` and is deliberately not committed: it can contain camera URLs, MQTT credentials, and authentication secrets.

## Updating configuration

1. Back up `/DATA/AppData/frigate/config/config.yaml`.
2. Edit it on the server.
3. Restart only the Frigate service:

   ```bash
   docker compose up -d
   ```

4. Confirm it is healthy with `docker ps` and `docker logs frigate`.

## Updating Frigate

Update the `image` tag in `compose.yaml`, then run `docker compose pull && docker compose up -d`.

Frigate stays containerized because its supported runtime includes FFmpeg, go2rtc, a process supervisor, shared memory, and direct GPU access.
