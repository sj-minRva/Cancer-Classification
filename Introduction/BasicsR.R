library(tidyverse)
data()
view(mpg)
?mpg
?mean
glimpse(mpg)
?filter


mpg_efficient <- filter(mpg, cty>=20)

