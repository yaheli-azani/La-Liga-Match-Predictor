import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

#---------------------------------------------------------
#CACHE the import of your heavy predictor module
#---------------------------------------------------------
@st.cache_resource  # caching decorator to avoid reruns
def load_predictor():
    """
    Loads the predictor module one time and caches it.
    """
    import la_liga_predictor
    return la_liga_predictor

predictor = load_predictor()

#---------------------------------------------------------
#Prepare team list for autocomplete
#---------------------------------------------------------
# Extract unique team names from historic matches
team_list = sorted(predictor.matches['HomeTeam'].unique())

#---------------------------------------------------------
#STREAMLIT PAGE LAYOUT
#---------------------------------------------------------
st.set_page_config(layout = "wide", page_title = "LaLiga Predictor")
st.title("LaLiga Future Game Predictor")
st.markdown("Enter match details in the sidebar and click **Predict** to view outcome chances.")

#---------------------------------------------------------
#SIDEBAR INPUTS
#---------------------------------------------------------
with st.sidebar.form("fixture_form"):
    st.header("Fixture Details")

    # Date picker widget
    date = st.date_input("Match Date", value = datetime.today())

    # Autocomplete/selectbox for team names
    home = st.selectbox("Home Team", options = team_list, index=0)
    away = st.selectbox("Away Team", options = team_list, index=1 if len(team_list) > 1 else 0)

    # Numeric inputs for odds (optional)
    b365h = st.number_input(
        "Home Odds (Horse Betting Odds e.g. 8/1 = 8.0)", value = 0.0, step = 0.001, format="%.3f"
    )
    b365d = st.number_input(
        "Draw Odds (Horse Betting Odds)", value = 0.0, step = 0.001, format = "%.3f"
    )
    b365a = st.number_input(
        "Away Odds (Horse Betting Odds", value = 0.0, step = 0.001, format = "%.3f"
    )

    submitted = st.form_submit_button("Predict")

#---------------------------------------------------------
#ON SUBMISSION: Validate & SHOW RESULTS
#---------------------------------------------------------
if submitted:
    # Check for odds provided
    odds_provided = any([b365h > 0, b365d > 0, b365a > 0])
    if not odds_provided:
        st.sidebar.warning(
            "No odds entered – default uniform probabilities (33%) will be used."
        )

    # Build fixture DataFrame
    fixture = pd.DataFrame([
        {
            "Date": date.strftime("%d/%m/%Y"),
            "HomeTeam": home,
            "AwayTeam": away,
            "B365H": float(b365h) if b365h > 0 else None,
            "B365D": float(b365d) if b365d > 0 else None,
            "B365A": float(b365a) if b365a > 0 else None
        }
    ])

    # Attempt prediction with error handling
    try:
        results = predictor.predict_future(fixture)

        # Layout: two columns for table and charts
        col1, col2 = st.columns([1, 1])

        # Prediction Table
        with col1:
            st.subheader("Prediction Table")
            st.table(
                results.set_index(["HomeTeam", "AwayTeam"])[
                    ["P(Home Win)", "P(Draw)", "P(Away Win)", "Prediction"]
                ]
            )

        # Charts
        with col2:
            probs = results.iloc[0][["P(Home Win)", "P(Draw)", "P(Away Win)"]]
            labels = ["Home Win", "Draw", "Away Win"]
            values = probs.tolist()

            st.subheader("Probability Bar Chart")
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(labels, values)
            ax_bar.set_ylabel("Probability")
            ax_bar.set_ylim(0, 1)
            ax_bar.set_title("Outcome Probabilities")
            st.pyplot(fig_bar)

            st.subheader("Probability Pie Chart")
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(values, labels=labels, autopct="%.1f%%", startangle=90)
            ax_pie.axis('equal')  # ensure circle
            st.pyplot(fig_pie)

    except KeyError as e:
        st.error(f"Team not found: {e}. Please select valid LaLiga team names.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")