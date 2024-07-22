#!/bin/sh

# eval "$(ssh-agent -s)"
# ssh-add ~/.ssh/id_rsa_ret

git pull git@github.com:Undefeated-man/RetrofittingLLM.git
git add -A .
git commit -m "update"
git push main --force
