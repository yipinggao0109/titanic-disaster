FROM rocker/r-base:4.3.1

WORKDIR /app

# Install R dependencies
COPY src/titanic_R/install_packages.R /app/install_packages.R
RUN Rscript install_packages.R

# Copy project files (including data)
COPY ../../src /app/src

# Run the R script
CMD ["Rscript", "src/titanic_R/titanic_main.R"]