# GitHub and Streamlit Community Cloud deployment

## Prepare the repository

Review the files before publishing. `.venv/`, SQLite database files, and generated processed data are ignored by Git. The source CSV is small and licensed CC BY 4.0; keep the attribution in `data/README.md`. Model pipelines and reports are included because they are project deliverables.

From the project root, initialize Git if needed, review the staged files, and push to a GitHub repository:

1. Run `git status` and confirm `.venv/` and `data/predictive_maintenance.db` are not listed.
2. If this folder is not already a Git repository, run `git init`.
3. Run `git add .` and `git status` to review the staged files.
4. Run `git commit -m "Build predictive maintenance forecasting project"`.
5. Create an empty GitHub repository in your account.
6. Add its URL with `git remote add origin https://github.com/<username>/<repository>.git`.
7. Run `git branch -M main` and `git push -u origin main`.

If `origin` already exists, inspect it with `git remote -v` rather than adding a duplicate.

## Deploy on Streamlit Community Cloud

1. Push the completed project to GitHub.
2. Open Streamlit Community Cloud and choose **Create app**.
3. Select the GitHub repository and `main` branch.
4. Set the app entrypoint to `app.py`.
5. In Advanced settings, select Python 3.13 to match the local environment. Community Cloud currently defaults to Python 3.12; the project supports Python 3.12 or newer.
6. Deploy and review the app logs if the build reports an error.

The root `requirements.txt` declares Python dependencies, and `.streamlit/config.toml` supplies the dark theme. Streamlit runs the app from the repository root, and its docs support entrypoint files in subdirectories when configuration remains at the root. [Streamlit file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization), [dependency configuration](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

## Database persistence

The application creates `data/predictive_maintenance.db` locally. Community Cloud does not guarantee that local file storage will persist, so the hosted database may be deleted. For a portfolio demonstration, this is acceptable if records are disposable. For durable records, replace the SQLite repository with a persistent hosted database and configure credentials through Streamlit secrets. [Streamlit data connection guidance](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data)

## Deployment scope

This app performs inference when an operator enters a reading or uploads a CSV. It does not connect to a PLC, MQTT broker, or streaming sensor gateway. A real industrial integration would require authenticated ingestion, device identity, monitoring, and a persistent database.
