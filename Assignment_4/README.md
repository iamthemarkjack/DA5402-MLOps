## News App using RSS Feed and MongoDB

## Setup Instructions

### 1. Running the Application

- **First-time setup**: Run the following command to build and start the containers:

    ```bash
    docker compose up -d --build

- **Subsequent runs**: Once the containers are built, run the application without rebuilding:

    ```bash
    docker compose up -d

### 2. Stopping the Application

- To stop the application, use the following command:

    ```bash
    docker compose down

- **Reinitialize the database**: If you want to remove the existing data and reinitialize the database, use the following command to delete the volume:

    ```bash
    docker compose down -v

### 3. Configuring the RSS Feed

- Go through the RSS XML structure of the news provider you wish to use.

- Update the relevant field paths in the **`docker-compose.yaml`** file with the correct paths to the RSS feed data.

- The path parser is flexible and handles both dictionary and list indexing. For example:

    - **`media_content/0/url`** is translated to **`entry_feed["media_content"][0]["url"]`**.

### 4. Ports Configuration

- The mongodb runs on port 27017 and the web application runs on port 5000 by default. You can change this in the **`docker-compose.yaml`** file if necessary.