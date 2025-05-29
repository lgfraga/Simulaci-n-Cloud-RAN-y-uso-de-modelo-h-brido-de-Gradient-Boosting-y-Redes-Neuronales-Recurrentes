#!/bin/bash
# Script para inicializar la configuración de red en vm3

# Verificar si el script se está ejecutando como root
if [ "$EUID" -ne 0 ]; then
    echo "Por favor, ejecuta este script como root (por ejemplo: sudo ./inicializar_red.sh)"
    exit 1
fi
echo "Reiniciando el servicio systemd-networkd..."
systemctl restart systemd-networkd.service

echo "Deshaciendo configuraciones previas..."
# Quitar sub-interfaces VLAN (por si quedaron)
ip link del enp0s3.110 2>/dev/null || true
ip link del enp0s3.112 2>/dev/null || true
ip link del enp0s3.113 2>/dev/null || true

# Reiniciar el bridge desde cero
ovs-vsctl --if-exists del-br ovs-br0

echo "Configurando el módulo 8021q..."
modprobe 8021q

echo "Creando y configurando el bridge OVS..."
ovs-vsctl add-br ovs-br0
ovs-vsctl add-port ovs-br0 enp0s3 vlan_mode=trunk trunk=110,112,113

echo "Configurando IP en el bridge..."
ip link set dev ovs-br0 up
ip addr add 192.168.1.70/24 dev ovs-br0
ip addr del 192.168.1.70/24 dev enp0s3 2>/dev/null || true

echo "Creando puertos internos para cada clase de tráfico..."
# Puerto interno para VLAN 110 (callout)
ovs-vsctl add-port ovs-br0 callout \
        -- set Interface callout type=internal \
        -- set Port callout tag=110

# Puerto interno para VLAN 112 (smsout)
ovs-vsctl add-port ovs-br0 smsout \
        -- set Interface smsout type=internal \
        -- set Port smsout tag=112

# Puerto interno para VLAN 113 (internet)
ovs-vsctl add-port ovs-br0 internet \
        -- set Interface internet type=internal \
        -- set Port internet tag=113

echo "Levantando puertos internos y asignando IPs..."
ip link set callout up
ip addr add 10.110.0.1/24 dev callout

ip link set smsout up
ip addr add 10.112.0.1/24 dev smsout

ip link set internet up
ip addr add 10.113.0.1/24 dev internet

echo "Configurando políticas por clase de tráfico (10 Mbit/s inicial)..."
ovs-vsctl set interface callout ingress_policing_rate=10000 ingress_policing_burst=1000
ovs-vsctl set interface smsout ingress_policing_rate=10000 ingress_policing_burst=1000
ovs-vsctl set interface internet ingress_policing_rate=10000 ingress_policing_burst=1000

echo "Iniciando servidores iperf3 en modo daemon..."
iperf3 -s -B 10.110.0.1 -p 9000 -i 1 --logfile /home/administrador/agenteSDN/iperf_servidor_callout.log --daemon
iperf3 -s -B 10.112.0.1 -p 9001 -i 1 --logfile /home/administrador/agenteSDN/iperf_servidor_smsout.log --daemon
iperf3 -s -B 10.113.0.1 -p 9002 -i 1 --logfile /home/administrador/agenteSDN/iperf_servidor_internet.log --daemon

echo "Configuración completada."


echo "Mostrando la configuración de red y el estado de los servicios..."
ip a
ovs-vsctl show
ovs-ofctl -O OpenFlow13 dump-flows ovs-br0
ovs-vsctl get interface callout ingress_policing_rate ingress_policing_burst
ovs-vsctl get interface smsout ingress_policing_rate ingress_policing_burst
ovs-vsctl get interface internet ingress_policing_rate ingress_policing_burst

