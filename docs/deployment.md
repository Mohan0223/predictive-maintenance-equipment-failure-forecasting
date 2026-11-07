# GitHub and Streamlit Community Cloud deployment

## GitHub repository

The public project repository is [divyansh5154/predictive-maintenance-equipment-failure-forecasting](https://github.com/divyansh5154/predictive-maintenance-equipment-failure-forecasting), on branch `main`. The project files, model artifacts, sample dataset, screenshots, and reports are included. `.venv/`, SQLite database files, and generated processed data are excluded by `.gitignore`. The source dataset is licensed CC BY 4.0; its attribution is in `data/README.md`.

For later code updates, from the project root run:

```powershell
git status
git add .
git commit -m "Describe the change"
git push
```

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with the GitHub account that has admin access to this repository.
2. Choose **Create app**, then **Yup, I have an app**.
3. Select repository `divyansh5154/predictive-maintenance-equipment-failure-forecasting`, branch `main`, and main file path `app.py`.
4. Optionally choose a custom app subdomain.
5. In Advanced settings, select Python 3.13 to match the local environment.
6. Choose **Deploy**. Community Cloud displays the shareable `*.streamlit.app` URL after the build succeeds.

The root `requirements.txt` declares Python dependencies, and `.streamlit/config.toml` supplies the dark theme. No Streamlit secrets are required for this demo. Streamlit reads the root configuration and runs the entrypoint from the repository root. See [Streamlit's deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization), and [dependency configuration](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Database persistence

The application creates `data/predictive_maintenance.db` locally. Community Cloud does not guarantee that local file storage will persist, so the hosted database may be deleted. For a portfolio demonstration, this is acceptable if records are disposable. For durable records, replace the SQLite repository with a persistent hosted database and configure credentials through Streamlit secrets. [Streamlit data connection guidance](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data)

## Deployment scope

This app performs inference when an operator enters a reading or uploads a CSV. It does not connect to a PLC, MQTT broker, or streaming sensor gateway. A real industrial integration would require authenticated ingestion, device identity, monitoring, and a persistent database.
