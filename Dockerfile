FROM allennlp/allennlp:v0.7.0

# Install requirements
COPY scripts/install_requirements.sh scripts/install_requirements.sh
COPY requirements.txt .
RUN ./scripts/install_requirements.sh

# Does not download models/data/Glove; assumes that they will be mounted

# Copy code
COPY obqa/ obqa/
COPY scripts/ scripts/
COPY training_config training_config/

ENTRYPOINT []
CMD ["/bin/bash"]