# EDA-TovTech-IMDb
## Dataset
Available [Here](https://www.kaggle.com/datasets/raedaddala/top-500-600-movies-of-each-year-from-1960-to-2024). Too larage to be uploaded to this repository, since it's one ~45MB .csv file with more than 60,000 rows and 24 columns, containing the top 500-600 films from each year between 1920 to 2025.
## Project Summary
For my final project, I wanted to explore something I'm passionate about, movies, through the lens of data science.
I started with a question: do directors who work across many genres tend to be more successful than those who stick to one type of film?
To answer this, I analyzed over 61,000 movies from IMDb's database.
First, I cleaned the messy data (converting things like "2.1K" votes to actual numbers and parsing director lists).
Then I created what I call a "success score" by combining both rating quality and audience reach, because I realized success isn't just about critical acclaim or popularity alone, it's both.
After engineering these features, I discovered something interesting: there's actually a small but statistically significant positive correlation (r=0.10) between how many genres a director works in and their overall success.
This led me to create "The Director's Compass," which groups directors into four categories based on their versatility and success levels.
While the correlation isn't huge, it suggests that being versatile doesn't hurt a director's career and might even help a little.
The whole process taught me how to take a real-world question and use data to find an actual, measurable answer.
## The Most Impressive Graph
<img width="1760" height="1320" alt="image" src="[https://github.com/IdanDalal/EDA/blob/main/plot4.jpg](https://raw.githubusercontent.com/IdanDalal/EDA-TovTech-IMDb/refs/heads/main/plot4.jpg)?raw=true" />
"The Director's Compass" plot shows 2,755 directors mapped by their versatility score on the x-axis and success score on the y-axis - both engineered percentiles, not raw numbers. The red-yellow-green color gradient shows performance on both axes simultaneously. The size of each director's dot represents the number of films they directed. The quadramts classify each director as an archetype: The Masters - top right, The Specialists - top left, The Craftsmen - bottom right, and The Explorers - bottom left.

## DataCamp Learning Testimonial
DataCamp completely changed how I see the world around me.
The courses steadily built up my confidence, understanding, and abilities by explaining functions and variables in a way that just clicked.
What really blew my mind was learning about "feature engineering": the idea that you can take existing data and transform it to reveal hidden patterns and create entirely new insights.
It made me realize that everything is data, and data is incredibly valuable if you know how to look at it.
This new perspective inspired my curiosity to explore data about my favorite subject: modern Hollywood films.
Before DataCamp, I would never have imagined I could answer complex questions about director success patterns using actual statistical evidence.
