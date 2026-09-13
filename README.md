# SmartRoute AI

An offline prototype for the AI-based smart car navigation system described in Review 1.

## Included pipeline

- Cyclic time feature engineering over the UCI Metro Interstate Traffic Volume dataset.
- Historical hour/weekend traffic forecasting with MAE, RMSE, and R2 evaluation.
- BPR congestion-aware edge travel times.
- Admissible A* routing with non-negative weather and incident penalties.
- Penalized alternative route generation.
- Streamlit dashboard with route comparison, corridor map, forecast quality, and traffic profile.
- Optional XGBoost traffic forecasting using weather and temporal features.
- Optional live OpenStreetMap topology through OSMnx with landmark-based endpoints.
- Intercity routing through OSRM for long journeys such as Mumbai to Pune.

The supplied dataset is a single I-94 sensor, so the dashboard uses a small deterministic Twin Cities corridor graph for a fully offline demonstration. The forecast is projected onto graph edges by capacity and road class; an OSMnx adapter can replace `build_demo_graph` when live topology is available.

## Repository structure

```text
AI project/
├── app.py                    # Streamlit dashboard entrypoint
├── mobile_app.py             # Alternate mobile-focused UI
├── api_server.py             # Optional FastAPI route service
├── traffic_analysis.py       # Core route and forecasting logic
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
├── tests/
│   ├── test_api_server.py
│   └── ...
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── cache/
```

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Click **New app**.
4. Select the repository and branch.
5. Set the app main file to `app.py`.
6. Click **Deploy**.

Optional: if you want to showcase the mobile version instead, set the main file to `mobile_app.py`.

## Notes

Enable **Use live OpenStreetMap graph** in the sidebar to download a real drivable road network. Enter landmark names such as `VIT University Main Gate` and `Katpadi Junction Railway Station`; the app geocodes them and finds the nearest roads. Leave live mode disabled for the instant offline demo. Live mode requires internet access and can take a minute to download the selected place.

For long-distance trips, select **Intercity / long travel**. Enter a broad search area such as `Mumbai and Pune, India`, then enter landmarks in the origin and destination cities. This mode uses the public OSRM driving router instead of downloading one enormous OSM graph.

For a dependency-light smoke test:

```powershell
python traffic_analysis.py
```