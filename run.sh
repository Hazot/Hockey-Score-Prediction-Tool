#!/bin/bash

# echo "TODO: fill in the docker run command"
docker run -d -p 5000:5000 --env COMET_API_KEY=$COMET_API_KEY -it ift6758/serving:1.0.0
