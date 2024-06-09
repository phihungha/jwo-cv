#!/usr/bin/env bash

poetry export -f requirements.txt > requirements.txt
grep 'git\+' requirements.txt > requirements-vcs.txt
grep 'git\+' -v requirements.txt > requirements-hashed.txt
