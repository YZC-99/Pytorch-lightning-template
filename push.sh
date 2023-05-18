#!/bin/bash
echo "Setting up Git user..."
git config --global user.name "yzc"
git config --global user.email "yzc@qq.com"

echo "Initializing Git repository..."
git init

echo "Adding files to Git repository..."
git rm -r --cached .
git add .
git commit -m "first commit"

git add ".gitignore"
git commit -m "Update .gitignore"


echo "Setting up access token..."
TOKEN="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"

echo "Building remote repository URL..."
REPO_URL="https://$TOKEN@github.com/YZC-99/Pytorch-lightning-template.git"
git push "$REPO_URL" main

echo "Script execution completed."