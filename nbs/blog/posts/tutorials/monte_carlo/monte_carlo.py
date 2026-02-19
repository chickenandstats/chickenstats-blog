import argparse
from typing import Literal

import polars as pl
import numpy as np
from scipy.stats import poisson
from pathlib import Path
import concurrent
from concurrent.futures import ThreadPoolExecutor

from chickenstats.chicken_nhl import Scraper, Season
from chickenstats.utilities import ChickenProgress

import datetime as dt


def aggregate_strength_states(team_stats: pl.DataFrame) -> pl.DataFrame:
    """Add a secondary strength state column to team stats data.

    Parameters:
        team_stats (pl.DataFrame):
            Polars dataframe of team statistics aggregated from the `chickenstats` library

    """
    # Create a new strength state column to combine powerplay and shorthanded
    even_strength = ["4v4", "3v3"]
    powerplay_list = ["5v4", "5v3", "4v3"]
    shorthanded_list = ["4v5", "3v4", "3v5"]

    team_stats = team_stats.with_columns(
        strength_state2=pl.when(pl.col("strength_state") == "5v5")
        .then(pl.lit("5v5"))
        .otherwise(
            pl.when(pl.col("strength_state").is_in(even_strength))
            .then(pl.lit("even_strength"))
            .otherwise(
                pl.when(pl.col("strength_state").is_in(powerplay_list))
                .then(pl.lit("powerplay"))
                .otherwise(
                    pl.when(pl.col("strength_state").is_in(shorthanded_list))
                    .then(pl.lit("shorthanded"))
                    .otherwise(pl.col("strength_state"))
                )
            )
        )
    )

    return team_stats


def augment_team_stats(team_stats: pl.DataFrame, level: Literal["team", "nhl"]) -> pl.DataFrame:
    """Aggregate to season stats for desired level, team or NHL."""
    if level == "team":
        group_columns = ["season", "session", "team", "is_home", "strength_state2"]
        non_agg_columns = ["game_id", "game_date", "opp_team", "strength_state"]

    elif level == "nhl":
        group_columns = ["season", "session", "is_home", "strength_state2"]
        non_agg_columns = ["team", "strength_state", "game_id", "game_date", "opp_team"]

    agg_stats = tuple(
        pl.col(x).sum()
        for x in team_stats.columns
        if x not in group_columns and x not in non_agg_columns and "p60" not in x and "percent" not in x
    )

    agg_stats = agg_stats + (pl.col("game_id").n_unique(),)

    team_stats = team_stats.group_by(group_columns).agg(agg_stats)

    team_stats = team_stats.with_columns(
        g_score_ax=pl.col("gf_adj") - pl.col("xgf_adj"), g_save_ax=pl.col("xga_adj") - pl.col("ga_adj")
    )

    team_stats = team_stats.with_columns(
        toi_gp=pl.col("toi") / pl.col("game_id"),
        gf_p60=pl.col("gf") / pl.col("toi") * 60,
        ga_p60=pl.col("ga") / pl.col("toi") * 60,
        gf_adj_p60=pl.col("gf_adj") / pl.col("toi") * 60,
        ga_adj_p60=pl.col("ga_adj") / pl.col("toi") * 60,
        xgf_p60=pl.col("xgf") / pl.col("toi") * 60,
        xga_p60=pl.col("xga") / pl.col("toi") * 60,
        xgf_adj_p60=pl.col("xgf_adj") / pl.col("toi") * 60,
        xga_adj_p60=pl.col("xga_adj") / pl.col("toi") * 60,
        g_score_ax_p60=pl.col("g_score_ax") / pl.col("toi") * 60,
        g_save_ax_p60=pl.col("g_save_ax") / pl.col("toi") * 60,
    )

    return team_stats


def prep_nhl_stats(team_stats: pl.DataFrame) -> pl.DataFrame:
    """Function to calculate the goals scored and allowed above the expected goals model, adjusted for score and venue.

    Parameters:
        team_stats (pl.DataFrame):
            Polars dataframe of team statistics aggregated from the `chickenstats` library


    """
    if "strength_state2" not in team_stats.columns:
        team_stats = aggregate_strength_states(team_stats)

    nhl_stats = augment_team_stats(team_stats=team_stats, level="nhl")

    return nhl_stats


def prep_team_stats(team_stats: pl.DataFrame, nhl_stats: pl.DataFrame, latest_date: str) -> pl.DataFrame:
    """Prepare team stats dataframe for later analysis. Nested within the prep today's function.

    Parameters:
        team_stats (pl.DataFrame):
            Polars dataframe of team statistics aggregated from the `chickenstats` library
        nhl_stats (pl.DataFrame):
            Polars dataframe of aggregated NHL-level statistics from the team stats dataframe at an earlier stage
        latest_date (str):
            Latest date to use in aggregating team stats - don't want later results to bias
            predictions of earlier games

    """
    if "strength_state2" not in team_stats.columns:
        team_stats = aggregate_strength_states(team_stats)

    latest_date_dt = dt.date(year=int(latest_date[:4]), month=int(latest_date[5:7]), day=int(latest_date[8:10]))
    team_stats = team_stats.filter(pl.col("game_date").str.to_date(format="%Y-%m-%d") < latest_date_dt)

    team_stats = augment_team_stats(team_stats=team_stats, level="team")

    # Adding mean NHL columns for the columns we'll use to predict the games
    predict_columns = [
        "xgf_p60",
        "xga_p60",
        "xgf_adj_p60",
        "xga_adj_p60",
        "gf_p60",
        "ga_p60",
        "gf_adj_p60",
        "ga_adj_p60",
        "g_score_ax_p60",
        "g_save_ax_p60",
        "toi_gp",
    ]

    for column in predict_columns:
        nhl_mean_map = dict(
            zip((nhl_stats["strength_state2"] + nhl_stats["is_home"].cast(pl.String)), nhl_stats[column], strict=False)
        )

        team_stats = team_stats.with_columns(
            (pl.col("strength_state2") + pl.col("is_home").cast(pl.String))
            .replace_strict(nhl_mean_map, return_dtype=pl.Float64, default=None)
            .alias(f"mean_{column}")
        )

    # Calculating the strength scores

    team_stats = team_stats.with_columns(
        off_strength=pl.col("xgf_adj_p60") / pl.col("mean_xgf_adj_p60"),
        def_strength=pl.col("xga_adj_p60") / pl.col("mean_xga_adj_p60"),
        toi_comparison=pl.col("toi_gp") / pl.col("mean_toi_gp"),
        scoring_strength=pl.col("g_score_ax_p60") / pl.col("mean_g_score_ax_p60"),
        goalie_strength=pl.col("g_save_ax_p60") / pl.col("mean_g_save_ax_p60"),
    )

    return team_stats


def prep_todays_games(
    schedule: pl.DataFrame, team_stats: pl.DataFrame, nhl_stats: pl.DataFrame, latest_date: str
) -> pl.DataFrame:
    """Function to prep today's games."""
    team_stats = prep_team_stats(team_stats=team_stats, nhl_stats=nhl_stats, latest_date=latest_date)

    todays_games = schedule.filter(pl.col("game_date") == latest_date)  # Filter the schedule for today's games

    venues = ["home", "away"]  # We need to account for venue effects on scorekeeping and performance
    strength_states = ["5v5", "powerplay", "shorthanded"]  # Segmenting by strength state

    team_value_dicts = {}  # Dictionary to store values to map to team stats columns

    for venue in venues:  # Looping through home and away for team stats
        for strength_state in strength_states:  # Looping through the strength states
            values = [  # These are the columns we need to append to the dataframe
                "off_strength",
                "def_strength",
                "scoring_strength",
                "goalie_strength",
                "toi_comparison",
            ]

            if strength_state == "powerplay":  # We don't care about defensive comparisons for powerplay
                remove_list = ["def_strength", "goalie_strength"]
                values = [x for x in values if x not in remove_list]

            elif strength_state == "shorthanded":  # We don't care about defensive comparisons for shorthanded
                remove_list = ["off_strength", "scoring_strength"]
                values = [x for x in values if x not in remove_list]

            for value in values:  # Need to loop through the new column values
                # Disaggregating the next couple of steps for readability

                if venue == "home":  # Getting a dummy value for filtering the dataframe
                    venue_dummy = 1

                else:
                    venue_dummy = 0

                dummy_replacements = {1: "home", 0: "away"}  # Dictionary to replace dummy variables later

                filter_conditions = (pl.col("strength_state2") == strength_state, pl.col("is_home") == venue_dummy)
                filter_df = team_stats.filter(filter_conditions)  # Filtering the dataframe

                # Getting a mapping column value for the nested dictionary, e.g., NAShome

                team_column = filter_df["team"]  # Getting the team names portion of the mapping key
                venue_column = filter_df["is_home"].replace_strict(
                    dummy_replacements, return_dtype=pl.String
                )  # Getting venue portion, but need to replace the dummy values

                mapping_column = team_column + venue_column  # Mapping column for nested dictionary, e.g., NAShome
                value_column = filter_df[value]  # Value column for nested dictionary
                nested_dictionary = dict(
                    zip(mapping_column, value_column, strict=False)
                )  # Nested dictionary being added to team_value_dicts

                dictionary_key = f"{venue}_{strength_state}_{value}"  # Key for the dictionary being added to team_value_dicts, e.g., home_5v5_off_strength

                team_value_dicts.update(
                    {dictionary_key: nested_dictionary}
                )  # Adding the nested dictionary to team_value_dicts

    predict_columns = [  # Columns to predict as part of the simulation
        "xgf_p60",
        "xga_p60",
        "xgf_adj_p60",
        "xga_adj_p60",
        "gf_p60",
        "ga_p60",
        "gf_adj_p60",
        "ga_adj_p60",
        "g_score_ax_p60",
        "g_save_ax_p60",
        "toi_gp",
    ]

    nhl_value_dicts = {}

    for venue in venues:  # Looping through home and away for mean NHL stats
        for strength_state in strength_states:  # Looping through the strength states
            for column in predict_columns:  # Need to loop through the new column values
                if venue == "home":  # Setting the integer for the dummy column
                    venue_dummy = 1

                else:
                    venue_dummy = 0

                # Getting the column to match on for the NHL stats
                filter_conditions = (
                    pl.col("strength_state2") == strength_state,
                    pl.col("is_home") == venue_dummy,
                )  # Splitting out for readability
                field_value = nhl_stats.filter(filter_conditions)[column][0]  # We need the value

                nhl_value_dicts.update(
                    {f"{venue}_{strength_state}_{column}": field_value}
                )  # Becomes part of the replace function later on

    add_columns = []  # Columns being added to the team stats dataframe e.g., off_strength_score, based on strength state and venue

    for (
        key,
        value,
    ) in team_value_dicts.items():  # Iterating through the dictionary created earlier to add them to the dataframe
        venue = f"{key[:4]}"  # This pulls in either "home" or "away" from the key, which is something like home_5v5_off_strength
        venue_team = f"{venue}_team"

        # Adds a new column with the values based on the team and strength state, with a new column name like home_5v5_off_strength
        new_column = (pl.col(venue_team) + venue).replace_strict(value, return_dtype=pl.Float64).alias(key)

        add_columns.append(new_column)  # Appending to the new columns list

    for key, value in nhl_value_dicts.items():  # Adding the NHL mean columns to the dataframe
        add_columns.append(
            pl.lit(value).alias(f"mean_{key}")
        )  # Adding the NHL mean column to the list to be added to the dataframe

    todays_games = todays_games.with_columns(add_columns)  # Adding the columns to the dataframe

    # Predicting home and away time on ice, goals, and expected goals based on historical performance and NHL means

    todays_games = todays_games.with_columns(
        predicted_home_5v5_toi=pl.col("home_5v5_toi_comparison")  # predicted home 5v5 TOI
        * pl.col("away_5v5_toi_comparison")
        * pl.col("mean_home_5v5_toi_gp"),
        predicted_home_powerplay_toi=pl.col("home_powerplay_toi_comparison")  # predicted home powerplay TOI
        * pl.col("away_shorthanded_toi_comparison")
        * pl.col("mean_home_powerplay_toi_gp"),
        predicted_home_shorthanded_toi=pl.col("home_shorthanded_toi_comparison")  # predicted home shorthanded TOI
        * pl.col("away_powerplay_toi_comparison")
        * pl.col("mean_home_shorthanded_toi_gp"),
        predicted_home_5v5_xgf_p60=pl.col("home_5v5_off_strength")
        * pl.col("away_5v5_def_strength")
        * pl.col("mean_home_5v5_xgf_adj_p60"),
        predicted_home_5v5_xga_p60=pl.col("home_5v5_def_strength")
        * pl.col("away_5v5_off_strength")
        * pl.col("mean_home_5v5_xga_adj_p60"),
        predicted_home_5v5_gf_p60=pl.col("home_5v5_off_strength")
        * pl.col("away_5v5_def_strength")
        * pl.col("home_5v5_scoring_strength")
        * pl.col("away_5v5_goalie_strength")
        * pl.col("mean_home_5v5_gf_adj_p60"),
        predicted_home_5v5_ga_p60=pl.col("home_5v5_def_strength")
        * pl.col("away_5v5_off_strength")
        * pl.col("home_5v5_goalie_strength")
        * pl.col("away_5v5_scoring_strength")
        * pl.col("mean_home_5v5_ga_adj_p60"),
        predicted_home_powerplay_xgf_p60=pl.col("home_powerplay_off_strength")
        * pl.col("away_shorthanded_def_strength")
        * pl.col("mean_home_powerplay_xgf_adj_p60"),
        predicted_home_powerplay_gf_p60=pl.col("home_powerplay_off_strength")
        * pl.col("away_shorthanded_def_strength")
        * pl.col("home_powerplay_scoring_strength")
        * pl.col("away_shorthanded_goalie_strength")
        * pl.col("mean_home_powerplay_gf_adj_p60"),
        predicted_home_shorthanded_xga_p60=pl.col("home_shorthanded_def_strength")
        * pl.col("away_powerplay_off_strength")
        * pl.col("mean_home_shorthanded_xga_adj_p60"),
        predicted_home_shorthanded_ga_p60=pl.col("home_shorthanded_def_strength")
        * pl.col("away_powerplay_off_strength")
        * pl.col("home_shorthanded_goalie_strength")
        * pl.col("away_powerplay_scoring_strength")
        * pl.col("mean_home_shorthanded_ga_adj_p60"),
        predicted_away_5v5_xgf_p60=pl.col("away_5v5_off_strength")
        * pl.col("home_5v5_def_strength")
        * pl.col("mean_away_5v5_xgf_adj_p60"),
        predicted_away_5v5_xga_p60=pl.col("away_5v5_def_strength")
        * pl.col("home_5v5_off_strength")
        * pl.col("mean_away_5v5_xga_adj_p60"),
        predicted_away_5v5_gf_p60=pl.col("away_5v5_off_strength")
        * pl.col("home_5v5_def_strength")
        * pl.col("away_5v5_scoring_strength")
        * pl.col("home_5v5_goalie_strength")
        * pl.col("mean_away_5v5_gf_adj_p60"),
        predicted_away_5v5_ga_p60=pl.col("away_5v5_def_strength")
        * pl.col("home_5v5_off_strength")
        * pl.col("away_5v5_goalie_strength")
        * pl.col("home_5v5_scoring_strength")
        * pl.col("mean_away_5v5_ga_adj_p60"),
        predicted_away_powerplay_xgf_p60=pl.col("away_powerplay_off_strength")
        * pl.col("home_shorthanded_def_strength")
        * pl.col("mean_away_powerplay_xgf_adj_p60"),
        predicted_away_powerplay_gf_p60=pl.col("away_powerplay_off_strength")
        * pl.col("home_shorthanded_def_strength")
        * pl.col("away_powerplay_scoring_strength")
        * pl.col("home_shorthanded_goalie_strength")
        * pl.col("mean_away_powerplay_gf_adj_p60"),
        predicted_away_shorthanded_xga_p60=pl.col("away_shorthanded_def_strength")
        * pl.col("home_powerplay_off_strength")
        * pl.col("mean_away_shorthanded_xga_adj_p60"),
        predicted_away_shorthanded_ga_p60=pl.col("away_shorthanded_def_strength")
        * pl.col("home_powerplay_off_strength")
        * pl.col("away_shorthanded_goalie_strength")
        * pl.col("home_powerplay_scoring_strength")
        * pl.col("mean_away_shorthanded_ga_adj_p60"),
    )

    return todays_games


def random_float() -> float:
    """Generate a random floating number between 0 and 1."""
    random_generator = np.random.default_rng()

    # return random_generator.triangular(left=0.0, mode=0.5, right=1.0)
    return random_generator.random()


def simulate_game(game: dict) -> dict:
    """Docstring."""
    prediction = {}

    home_5v5_toi = poisson.ppf(random_float(), game["predicted_home_5v5_toi"])
    home_pp_toi = poisson.ppf(random_float(), game["predicted_home_powerplay_toi"])
    home_sh_toi = poisson.ppf(random_float(), game["predicted_home_shorthanded_toi"])

    total_toi = home_5v5_toi + home_pp_toi + home_sh_toi

    if total_toi > 60:
        home_5v5_toi = home_5v5_toi - ((home_5v5_toi / total_toi) * (total_toi - 60))
        home_pp_toi = home_pp_toi - ((home_pp_toi / total_toi) * (total_toi - 60))
        home_sh_toi = home_sh_toi - ((home_sh_toi / total_toi) * (total_toi - 60))

    home_5v5_xgf_p60 = poisson.ppf(random_float(), game["predicted_home_5v5_xgf_p60"])
    home_5v5_gf_p60 = poisson.ppf(random_float(), game["predicted_home_5v5_gf_p60"])
    home_pp_xgf_p60 = poisson.ppf(random_float(), game["predicted_home_powerplay_xgf_p60"])
    home_pp_gf_p60 = poisson.ppf(random_float(), game["predicted_home_powerplay_gf_p60"])

    away_5v5_xgf_p60 = poisson.ppf(random_float(), game["predicted_away_5v5_xgf_p60"])
    away_5v5_gf_p60 = poisson.ppf(random_float(), game["predicted_away_5v5_gf_p60"])
    away_pp_xgf_p60 = poisson.ppf(random_float(), game["predicted_away_powerplay_xgf_p60"])
    away_pp_gf_p60 = poisson.ppf(random_float(), game["predicted_away_powerplay_gf_p60"])

    home_5v5_goals = home_5v5_xgf_p60 * (home_5v5_toi / 60)
    home_pp_goals = home_pp_xgf_p60 * (home_pp_toi / 60)
    home_total_goals = home_5v5_goals + home_pp_goals

    away_5v5_goals = away_5v5_xgf_p60 * (home_5v5_toi / 60)
    away_pp_goals = away_pp_xgf_p60 * (home_sh_toi / 60)
    away_total_goals = away_5v5_goals + away_pp_goals

    if home_total_goals > away_total_goals:
        home_win = 1
        away_win = 0
        draw = 0

    elif away_total_goals > home_total_goals:
        home_win = 0
        away_win = 1
        draw = 0

    else:
        home_win = 0
        away_win = 0
        draw = 1

    prediction.update(
        {
            "game_id": game["game_id"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "predicted_home_5v5_toi": home_5v5_toi,
            "predicted_home_powerplay_toi": home_pp_toi,
            "predicted_home_shorthanded_toi": home_sh_toi,
            "predicted_away_5v5_toi": home_5v5_toi,
            "predicted_away_powerplay_toi": home_sh_toi,
            "predicted_away_shorthanded_toi": home_pp_toi,
            "predicted_home_5v5_gf_p60": home_5v5_gf_p60,
            "predicted_home_5v5_xgf_p60": home_5v5_xgf_p60,
            "predicted_home_powerplay_gf_p60": home_pp_gf_p60,
            "predicted_home_powerplay_xgf_p60": home_pp_xgf_p60,
            "predicted_home_5v5_goals": home_5v5_goals,
            "predicted_home_powerplay_goals": home_pp_goals,
            "predicted_home_total_goals": home_total_goals,
            "predicted_away_5v5_gf_p60": away_5v5_gf_p60,
            "predicted_away_5v5_xgf_p60": away_5v5_xgf_p60,
            "predicted_away_powerplay_gf_p60": away_pp_gf_p60,
            "predicted_away_powerplay_xgf_p60": away_pp_xgf_p60,
            "predicted_away_5v5_goals": away_5v5_goals,
            "predicted_away_powerplay_goals": away_pp_goals,
            "predicted_away_total_goals": away_total_goals,
            "home_win": home_win,
            "away_win": away_win,
            "draw": draw,
        }
    )

    return prediction


def predict_game(
    game: dict,
    total_simulations: int = 10_000,
    disable_progress_bar: bool = False,
    save: bool = False,
    overwrite: bool = True,
) -> pl.DataFrame:
    """Predict game based on n number of simulations."""
    predictions = []

    with ChickenProgress(transient=True, disable=disable_progress_bar) as progress:
        pbar_message = f"Simulating {game['game_id']}..."
        simulation_task = progress.add_task(pbar_message, total=total_simulations)

        for sim_number in range(0, total_simulations):
            prediction = simulate_game(game=game)
            predictions.append(prediction)

            if sim_number == total_simulations - 1:
                pbar_message = f"Finished simulating {game['game_id']}"

            progress.update(simulation_task, description=pbar_message, advance=1, refresh=True)

    predictions = pl.DataFrame(predictions)

    if save:
        predictions_path = Path("./results/predictions.csv")

        if predictions_path.exists() and overwrite:
            saved_predictions = pl.read_csv(predictions_path, infer_schema_length=2000)

            predictions = pl.concat([saved_predictions, predictions], strict=False)

        predictions.write_csv(predictions_path)

    return predictions


def predict_games(
    predict_game_function,
    todays_games: pl.DataFrame,
    n_workers: int = 6,
    total_simulations: int = 10_000,
    disable_progress_bar: bool = False,
    save: bool = False,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Simulate today's games."""
    predictions_list = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(predict_game_function, game, total_simulations, disable_progress_bar, save, overwrite)
            for game in todays_games.to_dicts()
        ]

        for future in concurrent.futures.as_completed(futures):
            predictions_list.append(future.result())

    return pl.concat(predictions_list)


def process_winners(predictions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the predictions to "predict" a winner of the game."""
    group_list = ["game_id", "home_team", "away_team"]

    agg_stats_sum = ["home_win", "away_win", "draw"]
    agg_stats_mean = [x for x in predictions.columns if x not in group_list and x not in agg_stats_sum]

    agg_stats = [pl.col(x).sum().alias(f"predicted_{x}") for x in agg_stats_sum] + [
        pl.col(x).mean().alias(f"{x}_mean") for x in agg_stats_mean
    ]

    predicted_winners = predictions.group_by(group_list).agg(agg_stats)

    sum_stats = [f"predicted_{x}" for x in agg_stats_sum]
    add_stats = []
    for stat in agg_stats_sum:
        add_stat = (pl.col(f"predicted_{stat}") / pl.sum_horizontal(sum_stats)).alias(f"predicted_{stat}_percent")
        add_stats.append(add_stat)

    predicted_winners = predicted_winners.with_columns(add_stats)

    predicted_winners = predicted_winners.with_columns(
        predicted_winner=pl.when(pl.col("predicted_home_win_percent") > pl.col("predicted_away_win_percent"))
        .then(pl.col("home_team"))
        .otherwise(pl.col("away_team"))
    )

    columns = [
        "game_id",
        "home_team",
        "away_team",
        "predicted_winner",
        "predicted_home_win",
        "predicted_away_win",
        "predicted_draw",
        "predicted_home_win_percent",
        "predicted_away_win_percent",
        "predicted_draw_percent",
        "predicted_home_5v5_goals_mean",
        "predicted_home_powerplay_goals_mean",
        "predicted_home_total_goals_mean",
        "predicted_home_5v5_xgf_p60_mean",
        "predicted_home_powerplay_xgf_p60_mean",
        "predicted_home_5v5_toi_mean",
        "predicted_home_powerplay_toi_mean",
        "predicted_home_shorthanded_toi_mean",
        "predicted_away_5v5_goals_mean",
        "predicted_away_powerplay_goals_mean",
        "predicted_away_total_goals_mean",
        "predicted_away_5v5_xgf_p60_mean",
        "predicted_away_powerplay_xgf_p60_mean",
        "predicted_away_5v5_toi_mean",
        "predicted_away_powerplay_toi_mean",
        "predicted_away_shorthanded_toi_mean",
    ]

    predicted_winners = predicted_winners.select(columns)

    return predicted_winners


def assess_predictions(predicted_winners: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Takes the scheduled and checkes to see if the predicted results are corrected."""
    schedule = schedule.filter(pl.col("game_state") == "OFF")  # Getting only the finished games
    game_ids = schedule["game_id"].to_list()  # Taking the game IDs as a list

    home_win = pl.col("home_score") > pl.col("away_score")  # Condition to check if the home team won
    winners = schedule.select(
        pl.when(home_win).then(pl.col("home_team")).otherwise(pl.col("away_team")).alias("actual_winner")
    )["actual_winner"].to_list()  # Getting the winners as a list
    winners_dict = dict(zip(game_ids, winners, strict=False))  # Combining game IDs and winning teams as a dictionary

    predicted_winners = predicted_winners.with_columns(
        actual_winner=pl.col("game_id").replace_strict(winners_dict, default=None, return_dtype=pl.String)
    )

    predicted_winners = predicted_winners.with_columns(
        prediction_correct=pl.when(pl.col("actual_winner") == pl.col("predicted_winner"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )

    columns = [
        "game_id",
        "home_team",
        "away_team",
        "predicted_winner",
        "actual_winner",
        "prediction_correct",
        "predicted_home_win",
        "predicted_away_win",
        "predicted_draw",
        "predicted_home_win_percent",
        "predicted_away_win_percent",
        "predicted_draw_percent",
        "predicted_home_5v5_goals_mean",
        "predicted_home_powerplay_goals_mean",
        "predicted_home_total_goals_mean",
        "predicted_home_5v5_xgf_p60_mean",
        "predicted_home_powerplay_xgf_p60_mean",
        "predicted_home_5v5_toi_mean",
        "predicted_home_powerplay_toi_mean",
        "predicted_home_shorthanded_toi_mean",
        "predicted_away_5v5_goals_mean",
        "predicted_away_powerplay_goals_mean",
        "predicted_away_total_goals_mean",
        "predicted_away_5v5_xgf_p60_mean",
        "predicted_away_powerplay_xgf_p60_mean",
        "predicted_away_5v5_toi_mean",
        "predicted_away_powerplay_toi_mean",
        "predicted_away_shorthanded_toi_mean",
    ]

    assessed_predicted_winners = predicted_winners.select(columns)

    return assessed_predicted_winners


def main():
    """Function to run the Monte Carlo simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all-dates", help="Upload play-by-play data", action="store_true")
    parser.add_argument("--latest-date", help="Upload play-by-play data", action="store", type=str)
    parser.add_argument(
        "-s", "--simulations", help="Number of simulations per game to run", action="store", default="100_000", type=int
    )
    parser.add_argument(
        "-n",
        "--number-of-cores",
        help="Number of cores / pools for multiprocessing",
        action="store",
        default="6",
        type=int,
    )
    args = parser.parse_args()

    season = Season(2025, backend="polars")
    schedule = season.schedule(transient_progress_bar=True)

    conds = pl.col("game_state") == "OFF"
    game_ids = schedule.filter(conds)["game_id"].unique().to_list()

    data_directory = Path.cwd() / "data"
    stats_file = data_directory / "team_stats.csv"

    if not data_directory.exists():
        data_directory.mkdir()

    if stats_file.exists():
        team_stats = pl.read_csv(source=stats_file, infer_schema_length=2000)

        saved_game_ids = team_stats["game_id"].to_list()
        game_ids = [x for x in game_ids if x not in saved_game_ids]

    if game_ids:
        scraper = Scraper(game_ids, backend="polars", transient_progress_bar=True)
        # pbp = scraper.play_by_play
        scraped_team_stats = scraper.team_stats

        team_stats = pl.concat(
            [team_stats.with_columns(pl.col("bsf_adj_percent").cast(pl.Float64)), scraped_team_stats], strict=False
        )  # Quirk, don't ask
        team_stats.write_csv(stats_file)

    home_map = dict(zip(schedule["game_id"], schedule["home_team"], strict=False))

    team_stats = team_stats.with_columns(
        is_home=pl.when(pl.col("game_id").replace_strict(home_map, return_dtype=pl.String) == pl.col("team"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )

    nhl_stats = prep_nhl_stats(team_stats)

    if args.all_dates:
        results_directory = Path.cwd() / "results"
        predictions_file = results_directory / "predictions.csv"

        if predictions_file.exists():
            saved_predictions = pl.read_csv(predictions_file)
            conds = (
                pl.col("game_state") == "OFF",
                ~pl.col("game_id").is_in(saved_predictions["game_id"].unique().to_list()),
            )
            game_index = 0

        else:
            conds = pl.col("game_state") == "OFF"
            game_index = 75

        final_games = schedule.filter(conds).sort("game_id", descending=False)
        final_game_ids = final_games["game_id"].unique().to_list()[game_index:]

        dates = (
            schedule.filter(pl.col("game_id").is_in(final_game_ids))
            .sort("game_id", descending=False)["game_date"]
            .unique(maintain_order=True)
            .to_list()
        )

    elif args.latest_date:
        latest_date = dt.date(
            year=int(args.latest_date[:4]), month=int(args.latest_date[5:7]), day=int(args.latest_date[8:10])
        )

        results_directory = Path.cwd() / "results"
        predictions_file = results_directory / "predictions.csv"

        if predictions_file.exists():
            saved_predictions = pl.read_csv(predictions_file)
            conds = (
                pl.col("game_date").str.to_datetime(format="%Y-%m-%d") <= latest_date,
                ~pl.col("game_id").is_in(saved_predictions["game_id"].unique().to_list()),
                pl.col("game_id") > saved_predictions["game_id"].max(),
                pl.col("game_state") == "OFF",
            )

        else:
            conds = (
                pl.col("game_date").str.to_datetime(format="%Y-%m-%d") <= latest_date,
                pl.col("game_state") == "OFF",
            )

        final_games = schedule.filter(conds).sort("game_id", descending=False)
        final_game_ids = final_games["game_id"].unique().to_list()

        dates = (
            schedule.filter(pl.col("game_id").is_in(final_game_ids))
            .sort("game_id", descending=False)["game_date"]
            .unique(maintain_order=True)
            .to_list()
        )

    predictions_list = []

    with ChickenProgress() as progress:
        pbar_message = "Simulating games..."
        progress_task = progress.add_task(pbar_message, total=len(dates))

        for idx, date in enumerate(dates):
            todays_games = prep_todays_games(
                schedule=schedule, team_stats=team_stats, nhl_stats=nhl_stats, latest_date=date
            )

            today_game_ids = todays_games["game_id"].to_list()
            pbar_message = f"Simulating {date}: ({len(today_game_ids)} games)..."
            progress.update(progress_task, description=pbar_message, advance=False, refresh=True)

            predictions = predict_games(
                predict_game, todays_games, total_simulations=args.simulations, n_workers=args.number_of_cores
            )

            predictions_list.append(predictions)

            if idx == len(dates) - 1:
                pbar_message = "Finished simulating games"

            progress.update(progress_task, description=pbar_message, advance=1, refresh=True)

    predictions = pl.concat(predictions_list)
    predicted_winners = process_winners(predictions)
    assessed_predictions = assess_predictions(predicted_winners, schedule)

    results_directory = Path.cwd() / "results"
    predictions_file = results_directory / "predictions.csv"

    if predictions_file.exists():
        predictions = pl.concat([pl.read_csv(predictions_file), predictions])

    predictions.write_csv("./results/predictions.csv")

    assessed_predictions_file = results_directory / "assessed_predictions.csv"

    if assessed_predictions_file.exists():
        assessed_predictions = pl.concat([pl.read_csv(assessed_predictions_file), assessed_predictions])

    assessed_predictions.write_csv("./results/predicted_winners.csv")


if __name__ == "__main__":
    main()
