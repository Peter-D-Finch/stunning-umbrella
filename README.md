# stunning-umbrella
This is a hobby project that has two goals
1) Reading data from the BSC blockchain using web3 protocol (and an https endpoint) 
2) Building statistical tools to analyze the data collected

Currently the program pulls data from Pancakeswap's "Prediction". I was curious to see if there was any correlation between which way people were betting the price would move and which way the price actually moved.
![plot2](https://user-images.githubusercontent.com/71032947/151406204-8d59b281-9676-4833-b0ff-13d34930bb80.JPG)
Fig. 1. A scatter plot of % price change in the X-Axis and Bear-Bull-Ratio on the Y-Axis (The ratio of currency bet on a bullish outcome to bearish)
![plot1](https://user-images.githubusercontent.com/71032947/151406272-20bead78-2e79-4f58-ad5d-f02a7f01f699.JPG)
Fig. 2. A Gaussian Distribution fitted to Fig. 1.

So far the project has been able to give me insights such as:
  - The price movement direction matches the majority prediction ~57% of the time
  - There is a relationship between unpredictability of the price direction and the amount of liquidity
  - There is a relationship between volume and time of day
