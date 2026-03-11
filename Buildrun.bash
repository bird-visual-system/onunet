#!/bin/bash

# from lenovomic :
docker run -it -p 8887:8887 -e USER=$USER -e USERID=$UID \
 -v /home/michel:/home/jovyan \
 registry.gitlab.com/ptitmatheux/onunet

# from meteadvice :
#docker run -it -p 8888:8888 -e USER=$USER -e USERID=$UID -v /home/metea:/home/jovyan registry.gitlab.com/ptitmatheux/onunet
