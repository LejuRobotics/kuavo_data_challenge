#!/bin/bash

set -e

# 从环境变量获取必要的参数
GITHUB_REPO_URL="${opensource_project_origin_github}"
TEMP_REPO_PATH="${temp_target_repo_path_github}"
CURRENT_BRANCH="${CI_COMMIT_BRANCH}"
PROJECT_DIR="${CI_PROJECT_DIR}"
PROJECT_URL="${CI_PROJECT_URL}"
COMMIT_SHA="${CI_COMMIT_SHA}"
PIPELINE_ID="${CI_PIPELINE_ID}"
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
    retry_git_command "git -c core.sshCommand=\"ssh -i /home/gitlab-runner/.ssh/id_ed25519_data_challenge_isd -o IdentitiesOnly=yes -o StrictHostKeyChecking=no\" clone \"$GITHUB_REPO_URL\" git_repo"
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
  
  # Sync files to target repository, excluding .gitlab-ci.yml, ci_scripts, and .claude
  rsync -av --delete \
    --exclude='.git' \
    --exclude='.gitlab-ci.yml' \
    --exclude='ci_scripts' \
    --exclude='.claude' \
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

# Function to sync submodules from source repository
sync_submodules() {
  echo "Syncing submodules from source repository..."
  
  # Save current directory (TEMP_REPO_PATH)
  local temp_repo_dir="$TEMP_REPO_PATH/git_repo"
  
  # Go back to the original CI repository
  cd "$PROJECT_DIR"
  
  # Get submodule information for the current commit
  if [ -f .gitmodules ]; then
    echo "Found .gitmodules in source repository"

    # Using .gitmodules and gitlinks to get submodule info without cloning
    
    # Get list of submodules with their paths and URLs
    git config --file .gitmodules --get-regexp path | while read -r key path; do
      # Extract module name from the key
      module_name=$(echo "$key" | sed 's/^submodule\.\(.*\)\.path$/\1/')
      
      # Get the URL for this submodule from .gitmodules
      url=$(git config --file .gitmodules --get "submodule.${module_name}.url")
      
      # Get the commit hash for this submodule using gitlink (without cloning)
      # git ls-tree shows the gitlink which contains the exact commit SHA
      commit_hash=$(git ls-tree HEAD "$path" | awk '{print $3}')
      
      echo "Processing submodule: $module_name"
      echo "  Path: $path"
      echo "  URL: $url"
      echo "  Commit: $commit_hash"
      
      # Switch to temp repository
      cd "$temp_repo_dir"
      
      # Check if the path already exists (could be a file, directory, or submodule)
      if [ -e "$path" ] || [ -d "$path" ]; then
        echo "  Path $path already exists, cleaning it up..."
        # Remove from git index if it's tracked
        git rm -rf --cached "$path" 2>/dev/null || true
        # Remove physical files/directories
        rm -rf "$path"
      fi
      
      # Check if submodule is already registered in .gitmodules
      if git config --file .gitmodules --get "submodule.${module_name}.url" >/dev/null 2>&1; then
        echo "  Submodule $module_name already registered, removing from config..."
        git config --file .gitmodules --remove-section "submodule.${module_name}" 2>/dev/null || true
        git config --remove-section "submodule.${module_name}" 2>/dev/null || true
      fi
      
      # Add the submodule
      echo "  Adding submodule to target repository..."
      git submodule add -f "$url" "$path"
      
      # Checkout the specific commit
      if [ -n "$commit_hash" ]; then
        cd "$path"
        git checkout "$commit_hash"
        cd "$temp_repo_dir"
      fi
      
      # Go back to source repository for next iteration
      cd "$PROJECT_DIR"
    done
    
    # Return to temp repository
    cd "$temp_repo_dir"
    echo "Submodules synchronization completed"
  else
    echo "No .gitmodules file found in source repository"
    cd "$temp_repo_dir"
  fi
  
  return 0
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
    
    # If master branch, push to main branch on remote; otherwise push to the same branch name
    if [ "$CURRENT_BRANCH" = "master" ]; then
      echo "Master branch detected, pushing to main branch on remote"
      retry_git_command "git -c core.sshCommand=\"ssh -i /home/gitlab-runner/.ssh/id_ed25519_data_challenge_isd -o IdentitiesOnly=yes -o StrictHostKeyChecking=no\" push -f origin master:main"
    else
      retry_git_command "git -c core.sshCommand=\"ssh -i /home/gitlab-runner/.ssh/id_ed25519_data_challenge_isd -o IdentitiesOnly=yes -o StrictHostKeyChecking=no\" push -f origin $CURRENT_BRANCH"
    fi
    
    echo "Changes pushed successfully"
    
    return 0
  else
    echo "No changes to commit"
    return 0
  fi
}

# Function to send failure notification
send_failure_notification() {
  local error_msg="$1"
  if [ -n "$WECHAT_TOKEN" ]; then
    local pipeline_url="https://www.lejuhub.com/robotembodieddata/kuavo_data_challenge/-/pipelines/${PIPELINE_ID}"
    bash "$PROJECT_DIR/ci_scripts/wechat_bot_notify.sh" \
      "kuavo_data_challenge GitHub sync for branch $CURRENT_BRANCH failed: $error_msg. Pipeline: $pipeline_url" \
      "$WECHAT_TOKEN"
  fi
}

# Function to cleanup
cleanup() {
  echo "Cleaning up..."
  cd "$PROJECT_DIR"
  git clean -dfx
  git checkout .
}

# Main execution with error handling
main() {
  echo "Starting GitHub sync process..."
  
  # Setup target repository
  if ! setup_target_repository; then
    send_failure_notification "Failed to setup target repository"
    exit 1
  fi
  
  # Sync files
  if ! sync_files; then
    send_failure_notification "Failed to sync files"
    exit 1
  fi
  
  # Remove large files
  if ! remove_large_files; then
    send_failure_notification "Failed to remove large files"
    exit 1
  fi
  
  # Sync submodules from source repository
  if ! sync_submodules; then
    send_failure_notification "Failed to sync submodules"
    exit 1
  fi
  
  # Commit and push
  if ! commit_and_push; then
    send_failure_notification "Failed to commit and push changes"
    exit 1
  fi
  
  # Cleanup
  cleanup
  
  # Send success notification with pipeline URL
  if [ -n "$WECHAT_TOKEN" ]; then
    local pipeline_url="https://www.lejuhub.com/robotembodieddata/kuavo_data_challenge/-/pipelines/${PIPELINE_ID}"
    bash "$PROJECT_DIR/ci_scripts/wechat_bot_notify.sh" \
      "kuavo_data_challenge GitHub sync for branch $CURRENT_BRANCH succeeded. Pipeline: $pipeline_url" \
      "$WECHAT_TOKEN"
  fi
  
  echo "GitHub sync process completed successfully!"
}

# Run main function with error handling
if ! main "$@"; then
  echo "GitHub sync process failed!"
  exit 1
fi
