For installation of NATS Jetstream:

1. Install Docker Desktop
2. Download the nats.conf file in this directory
3. Execute "docker run -d --name nats-jetstream -p 4222:4222 -p 8222:8222 -v [your local directory]/nats.conf nats:latest -c /etc/nats/nats.conf"
