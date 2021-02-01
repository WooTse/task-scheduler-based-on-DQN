#!/bin/bash

#Pi Cluster Node SSH Public Key Assign Tool V0.1
#Created by Yuan Wang <bg3mdo@gmail.com>
#
#Copyright (C) 2019 by Yuan Wang BG3MDO
#
#This library is free software; you can redistribute it and/or
#modify it under the terms of the GNU Library General Public
#License as published by the Free Software Foundation; either
#version 2 of the License, or (at your option) any later version.
#
#This library is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#Library General Public License for more details.
#
#You should have received a copy of the GNU Library General Public
#License along with this library; if not, write to the
#Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
#Boston, MA  02110-1301, USA.

echo "Pi Cluster Node SSH Public Key Assign Tool V0.1"
echo "By Yuan Wang <bg3mdo@gmail.com>"
echo ""

ip="192.168.100"
start_ip=100
end_ip=123
useid="pi"

cmd="ssh-copy-id"

while [ $start_ip -le $end_ip ]; do
	cmdtemp="$cmd $useid@$ip.$start_ip"
	echo $cmdtemp
	eval $cmdtemp
	start_ip=$(($start_ip + 1)) 
done