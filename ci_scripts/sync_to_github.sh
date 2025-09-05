#!/bin/bash

set -e

# 从环境变量获取必要的参数
GITHUB_REPO_URL="${opensource_project_origin_github}"
TEMP_REPO_PATH="${temp_target_repo_path_github}"
CURRENT_BRANCH="${CI_COMMIT_BRANCH}"
PROJECT_DIR="${CI_PROJECT_DIR}"
PROJECT_URL="${CI_PROJECT_URL}"
COMMIT_SHA="${CI_COMMIT_SHA}"
WECHAT_TOKEN="${WECHAT_BOT_TOKEN}"

# Retry function with exponential backoff
retry_git_command() {
  local cmd="$1"
  local max_attempts=10
  local attempt=1
  
  while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt/$max_attempts: $cmd"
    if eval "$cmd"; then
      echo "Command succeeded on attempt $attempt"
      return 0
    else
      if [ $attempt -eq $max_attempts ]; then
        echo "Command failed after $max_attempts attempts"
        return 1
      fi
      local delay=$((2**attempt))
      echo "Command failed, retrying in $delay seconds..."
      sleep $delay
      attempt=$((attempt + 1))
    fi
  done
}

# Function to setup target repository
setup_target_repository() {
  echo "Setting up target repository..."
  mkdir -p "$TEMP_REPO_PATH"
  echo "Created/verified directory: $TEMP_REPO_PATH"
  cd "$TEMP_REPO_PATH"
  echo "Changed to directory: $(pwd)"
  
  # Clone repository if not exists
  if [ ! -d "git_repo/.git" ]; then
    git -c core.sshCommand="ssh -i /home/gitlab-runner/.ssh/id_ed25519_data_challenge_isd -o IdentitiesOnly=yes -o StrictHostKeyChecking=no" \
      clone "$GITHUB_REPO_URL" git_repo || echo "After clone github"
  fi
  
  cd git_repo
  retry_git_command "git -c core.sshCommand=\"ssh -i /home/gitlab-runner/.ssh/id_ed25519_data_challenge_isd -o IdentitiesOnly=yes -o StrictHostKeyChecking=no\" fetch origin"
  
  # Handle branch creation/checkout
  if git show-ref --verify --quiet refs/remotes/origin/"$CURRENT_BRANCH"; then
    echo "Branch $CURRENT_BRANCH exists on remote. Checking out."
    git checkout -B "$CURRENT_BRANCH" origin/"$CURRENT_BRANCH"
  else
    echo "Branch $CURRENT_BRANCH does not exist on remote. Creating new branch."
    if git show-ref --verify --quiet refs/heads/"$CURRENT_BRANCH"; then
      echo "Local branch $CURRENT_BRANCH exists. Checking out and resetting."
      git checkout "$CURRENT_BRANCH"
      git reset --hard HEAD
    else
      echo "Creating new local branch $CURRENT_BRANCH."
      if git show-ref --verify --quiet refs/remotes/origin/main; then
        echo "Using origin/main as base branch"
        git checkout -b "$CURRENT_BRANCH" origin/main
      elif git show-ref --verify --quiet refs/remotes/origin/master; then
        echo "Using origin/master as base branch"
        git checkout -b "$CURRENT_BRANCH" origin/master
      else
        echo "No priority base branch found, creating from HEAD"
        git checkout -b "$CURRENT_BRANCH"
      fi
    fi
  fi
  
  # Reset to remote branch if exists
  git reset --hard origin/"$CURRENT_BRANCH" 2>/dev/null || echo "Branch doesn't exist on remote yet"
  git clean -df
}

# Function to sync files
sync_files() {
  echo "Syncing files from source to target repository..."
  cd "$PROJECT_DIR"
  
  # Sync files to target repository, excluding .gitlab-ci.yml and ci_scripts
  rsync -av --delete \
    --exclude='.git' \
    --exclude='.gitlab-ci.yml' \
    --exclude='ci_scripts' \
    . "$TEMP_REPO_PATH/git_repo"
}

# Function to remove large files
remove_large_files() {
  echo "Checking for files larger than 80MB (GitHub limit)..."
  cd "$TEMP_REPO_PATH/git_repo"
  
  large_files=$(find . -type f -size +80M ! -path "./.git/*" 2>/dev/null)
  
  if [ -n "$large_files" ]; then
    echo "Found files larger than 80MB that will be removed:"
    echo "$large_files"
    echo "$large_files" | while read -r file; do
      if [ -f "$file" ]; then
        file_size=$(du -h "$file" | cut -f1)
        echo "Removing large file: $file (${file_size})"
        rm -f "$file"
      fi
    done
    echo "Completed removal of large files."
  else
    echo "No files larger than 80MB found."
  fi
}

# Function to commit and push changes
commit_and_push() {
  echo "Committing and pushing changes..."
  cd "$TEMP_REPO_PATH/git_repo"
  
  # Add all changes
  git add . && git add -u
  
  # Check if there are changes to commit
  if [ -n "$(git status --porcelain)" ]; then
    git commit -am "sync ${PROJECT_URL}/commit/${COMMIT_SHA}"
    retry_git_command "git -c core.sshCommand=\"ssh -i /home/gitlab-runner/.ssh/id_ed25519_data_challenge_isd -o IdentitiesOnly=yes -o StrictHostKeyChecking=no\" push -f origin $CURRENT_BRANCH"
    
    # If master branch, also push to main branch
    if [ "$CURRENT_BRANCH" = "master" ]; then
      echo "Master branch detected, also pushing to main branch"
      retry_git_command "git -c core.sshCommand=\"ssh -i /home/gitlab-runner/.ssh/id_ed25519_data_challenge_isd -o IdentitiesOnly=yes -o StrictHostKeyChecking=no\" push -f origin HEAD:main"
    fi
    
    # Send success notification
    if [ -n "$WECHAT_TOKEN" ]; then
      bash "$PROJECT_DIR/ci_scripts/wechat_bot_notify.sh" \
        "kuavo_data_challenge GitHub sync for branch $CURRENT_BRANCH succeeded" \
        "$WECHAT_TOKEN"
    fi
    
    return 0
  else
    echo "No changes to commit"
    return 0
  fi
}

# Function to cleanup
cleanup() {
  echo "Cleaning up..."
  cd "$PROJECT_DIR"
  git clean -dfx
  git checkout .
}

# Main execution
main() {
  echo "Starting GitHub sync process..."
  
  # Setup target repository
  setup_target_repository
  
  # Sync files
  sync_files
  
  # Remove large files
  remove_large_files
  
  # Commit and push
  commit_and_push
  
  # Cleanup
  cleanup
  
  echo "GitHub sync process completed successfully!"
}

# Run main function
main "$@"
