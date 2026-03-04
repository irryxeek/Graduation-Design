#!/bin/bash
cat $1 |while read -r line || [[ -n "$line" ]];
do
filename=$(echo "$line" | sed -E 's#.*/([^?]+).*#\1#')
echo "$filename"
wget $line -O $2/$filename
done