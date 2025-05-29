#!/bin/bash
# Script para configurar VLANs en vm1

# Verificar si el script se está ejecutando como root
if [ "$EUID" -ne 0 ]; then
    echo "Por favor, ejecuta este script como root (por ejemplo: sudo ./configurar_vlan_vm1.sh)"
    exit 1
fi

echo "Configurando módulo 8021q..."
modprobe 8021q

echo "Creando sub-interfaces VLAN..."
ip link add link enp0s8 name enp0s8.110 type vlan id 110
ip link add link enp0s9 name enp0s9.112 type vlan id 112
ip link add link enp0s10 name enp0s10.113 type vlan id 113

echo "Levantando interfaces..."
ip link set enp0s8.110 up
ip link set enp0s9.112 up
ip link set enp0s10.113 up

echo "Asignando IPs..."
ip addr add 10.110.0.2/24 dev enp0s8.110
ip addr add 10.112.0.2/24 dev enp0s9.112
ip addr add 10.113.0.2/24 dev enp0s10.113

echo "Configuración de VLANs completada."
