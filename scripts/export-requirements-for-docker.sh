#!/usr/bin/env bash

grep 'git\+' requirements.txt > requirements-vcs.txt
grep 'git\+' -v requirements.txt > requirements-hashed.txt
