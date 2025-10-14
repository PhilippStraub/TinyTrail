#!/bin/bash
# Setup SSH keys for passwordless access to Pi

PI_IP="$1"
PI_USER="${2:-pi}"

if [ -z "$PI_IP" ]; then
    echo "Usage: $0 <pi_ip> [pi_user]"
    exit 1
fi

echo "Setting up SSH key authentication for $PI_USER@$PI_IP"

# Check if default SSH key exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Creating SSH key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" -C "$(whoami)@$(hostname)"
fi

echo "Copying SSH key to Pi..."
echo "You will need to enter the Pi password one more time:"

# Copy SSH key to Pi
ssh-copy-id -o ConnectTimeout=5 -o StrictHostKeyChecking=no $PI_USER@$PI_IP

if [ $? -eq 0 ]; then
    echo "✓ SSH key authentication setup complete!"
    echo "Testing passwordless connection..."
    
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $PI_USER@$PI_IP "echo 'SSH key authentication working!'" 2>/dev/null; then
        echo "✓ Passwordless SSH connection confirmed!"
    else
        echo "⚠ SSH key copy succeeded but connection test failed"
        exit 1
    fi
else
    echo "✗ SSH key setup failed"
    exit 1
fi