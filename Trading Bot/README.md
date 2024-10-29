# Trading Bot
This folder contains the most relevant scripts and features a set of six strategies applied to specific time frame data.

For more information, refer to the README in the strategies folder.

# Objectives
The goal of this folder is to implement a flexible backtest script that can compute various backtesting scenarios. The backtest script will prompt several questions, including:

- Which broker do you want for data?
- What granularity?
- Which derivatives?

The script is designed to return a backtest saved in HTML format and print several key metrics, such as Max Drawdown, ROI, and Buy and Hold return.

# Automated Bot
The automated bot can be executed with an Interactive Brokers demo account. Note that specific requirements may be necessary for proper usage, such as a tailored IP address and socket port.

# Trading Management Rules
To streamline operations, most strategies and the bot follow specific trading management rules:

Every order placed has a fixed stop and limit order (for stop loss and take profit) upon execution.
Only one order is placed at a time to simplify portfolio management.
The stop loss and take profit distances from the current price are determined using two coefficients: TPSL_ratio and SL_coef. These parameters can be optimized for backtesting, and the bot is programmed to optimize its own trade management parameters once a week.