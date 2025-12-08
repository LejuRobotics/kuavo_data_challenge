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

  local temp_repo_dir="$TEMP_REPO_PATH/git_repo"

  cd "$PROJECT_DIR"

  if [ ! -f .gitmodules ]; then
    echo "No .gitmodules file found in source repository"
    cd "$temp_repo_dir"
    return 0
  fi

  echo "Found .gitmodules in source repository"

  # Collect all submodule information first to avoid subshell issues with pipe
  # Using arrays to store submodule info
  local -a submodule_names=()
  local -a submodule_paths=()
  local -a submodule_urls=()
  local -a submodule_commits=()

  # Use process substitution instead of pipe to avoid subshell
  while IFS= read -r line; do
    if [[ "$line" =~ submodule\.(.+)\.path\ (.+) ]]; then
      local module_name="${BASH_REMATCH[1]}"
      local path="${BASH_REMATCH[2]}"
      local url=$(git config --file .gitmodules --get "submodule.${module_name}.url")
      local commit_hash=$(git ls-tree HEAD "$path" 2>/dev/null | awk '{print $3}')

      submodule_names+=("$module_name")
      submodule_paths+=("$path")
      submodule_urls+=("$url")
      submodule_commits+=("$commit_hash")

      echo "Found submodule: $module_name"
      echo "  Path: $path"
      echo "  URL: $url"
      echo "  Commit: $commit_hash"
    fi
  done < <(git config --file .gitmodules --get-regexp path)

  # Check if we found any submodules
  if [ ${#submodule_names[@]} -eq 0 ]; then
    echo "No submodules found in .gitmodules"
    cd "$temp_repo_dir"
    return 0
  fi

  echo "Found ${#submodule_names[@]} submodule(s) to process"

  # Now process each submodule
  for i in "${!submodule_names[@]}"; do
    local module_name="${submodule_names[$i]}"
    local path="${submodule_paths[$i]}"
    local url="${submodule_urls[$i]}"
    local commit_hash="${submodule_commits[$i]}"

    echo ""
    echo "Processing submodule [$((i+1))/${#submodule_names[@]}]: $module_name"
    echo "  Path: $path"
    echo "  URL: $url"
    echo "  Target commit: $commit_hash"

    # Switch to temp repository
    cd "$temp_repo_dir"

    # Clean up existing path (could be file, directory, or submodule from rsync)
    if [ -e "$path" ] || [ -d "$path" ]; then
      echo "  Cleaning up existing path: $path"
      git rm -rf --cached "$path" 2>/dev/null || true
      rm -rf "$path"
    fi

    # Clean up submodule config if already registered
    if git config --file .gitmodules --get "submodule.${module_name}.url" >/dev/null 2>&1; then
      echo "  Removing existing submodule config for: $module_name"
      git config --file .gitmodules --remove-section "submodule.${module_name}" 2>/dev/null || true
      git config --remove-section "submodule.${module_name}" 2>/dev/null || true
    fi

    # Also check .git/config for stale submodule entries
    if git config --get "submodule.${module_name}.url" >/dev/null 2>&1; then
      echo "  Removing stale submodule entry from .git/config: $module_name"
      git config --remove-section "submodule.${module_name}" 2>/dev/null || true
    fi

    # Add the submodule
    echo "  Adding submodule: $url -> $path"
    if ! git submodule add -f "$url" "$path"; then
      echo "  ERROR: Failed to add submodule $module_name"
      continue
    fi

    # Checkout the specific commit
    if [ -n "$commit_hash" ]; then
      echo "  Checking out specific commit: $commit_hash"
      cd "$path"

      # Fetch to ensure we have the commit (in case of shallow clone)
      git fetch origin 2>/dev/null || echo "  Warning: fetch failed, trying checkout anyway"

      if git checkout "$commit_hash"; then
        echo "  Successfully checked out commit: $commit_hash"
      else
        echo "  ERROR: Failed to checkout commit $commit_hash for $module_name"
        echo "  Current HEAD: $(git rev-parse HEAD)"
      fi

      cd "$temp_repo_dir"
    else
      echo "  WARNING: No commit hash found for $module_name, using default branch"
    fi

    # Stage the submodule changes
    git add "$path"
    git add .gitmodules
  done

  cd "$temp_repo_dir"
  echo ""
  echo "Submodules synchronization completed"
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
