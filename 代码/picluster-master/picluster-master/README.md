# picluster

# Pi Cluster Image Deployment Tools Repository

**ifconfgen** is used to generate ip configure and hostname files for Raspberry Pi, in Python.

**ledservice** is a debug LED control program, Pi HAT schematic is under schematic, in C for speed optimisation. 

**mtfclone** is a Raspberry Pi image to multiple TF cards clone tool - batch clone work, in shell script.

**pishrink** is used to shrink Raspberry Pi image size - removing blank segments in a image file - reduce time for clone work, in shell script.

**dumptf2img** is used to dump a TF card to TF card image file, in shell script, then the image can be compressed by **pishrink** for smaller size.

**ipconf2tf** is an automatic IP/Hostname preparation tool for individual TF card, in Python.

**schematic** contains a Raspberry Pi power supply HAT schematic, in PDF format. 

**sshkeymap** is used to assign a master SSH public key to cluster node, then master no password needed to log in nodes, in shell script. 

By **Yuan Wang BG3MDO**, Under GNU GPL.
