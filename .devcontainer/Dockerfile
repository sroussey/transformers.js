# Base image with Node.js
FROM mcr.microsoft.com/devcontainers/typescript-node:24-bookworm

# Install Bun as the node user
RUN su - node -c "curl -fsSL https://bun.sh/install | bash"

# Set environment variables for Bun
ENV BUN_INSTALL="/home/node/.bun" 
ENV PATH="$BUN_INSTALL/bin:$PATH" 

RUN /home/node/.bun/bin/bunx playwright install-deps

# Set zsh as default shell for the existing non-root user (node in this image)
ARG USERNAME=node
RUN if id -u "$USERNAME" >/dev/null 2>&1; then chsh -s "$(which zsh)" "$USERNAME"; else echo "user $USERNAME not found"; fi

# Default shell
CMD [ "zsh" ]
