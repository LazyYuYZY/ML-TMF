#!/usr/bin/python

import os
import sys
import json

if os.getuid() !=0:
    print """
ERROR: This script requires root privileges. 
       Use 'sudo' to run it.
"""
    quit()

from scapy.all import *

try:
    iface = sys.argv[1]
except:
    iface="veth1"

# MPLS layer addition to scapy
class MPLS(Packet):
        name = "MPLS"
        fields_desc =  [
                BitField("label", 3, 20),
                BitField("experimental_bits", 0, 3),
                BitField("bottom_of_label_stack", 1, 1),
                ByteField("TTL", 255)
        ]
 
def sendmpls(SrcIP,DstIP,SrcPort,DstPort,Protocol):
    print "Sending IP packet to " + DstIP
    p = (Ether(dst="90:e2:ba:8e:24:2d", src="90:e2:ba:8e:24:2c", type=0x8847)/
        MPLS(label=0x0000a)/
        IP(src=SrcIP, dst=DstIP)/
        UDP(sport=SrcPort,dport=DstPort)/
        "This is a test")
    sendp(p, iface=iface) 

# sendmpls("10.0.0.1","10.0.0.2",7,7,0)

f = open("dat.json")
line = f.readline()
while line:
    s = json.loads(line)
    SrcIP = s["SrcIP"]
    DstIP = s["DstIP"]
    SrcPort = s["SrcPort"]
    DstPort = s["DstPort"]
    Protocol = s["Protocol"]
    sendmpls(SrcIP,DstIP,SrcPort,DstPort,Protocol)
    line = f.readline()
f.close()