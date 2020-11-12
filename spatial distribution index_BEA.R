
# Pre-Processing #
#1. Download the 'CAINC5S' dataset under 'Personal Income' (State and Local)
#     https://apps.bea.gov/regional/downloadzip.cfm
#2. Remove the quotations from the GeoFIPS column...this can be done in excel

library(readr)
#county <- read_csv("D:/thor's folder/grad stuff/Transpo/Transport Econ/data/BEA/clustering/CAINC5S__ALL_AREAS_1969_2000x.csv")
county <- read_csv("D:/thor's folder/grad stuff/Transpo/Transport Econ/data/BEA/clustering/CAINC5S_1969_2000_ALL_AREAS_.csv")

#removes national, state, and regions
county = county[-which(county$GeoFIPS%%1000==0),]
county <- county[-which(is.na(county$Description)),]

#ensures data is read as numeric
cols.num <- c(grep('1969',colnames(county)):grep('2000',colnames(county)))
county[cols.num] <- sapply(county[cols.num],as.numeric)

#m will be filled with the spatialGINI for each industry for each year
m <- data.frame()
farm <- data.frame(1:length(unique(county$GeoFIPS)))
for(d in unique(county$Description)){
  sect <- county[which(county$Description==d),]
  g=c()
  g[1] = d
  for(t in 1969:2000){
    f = as.numeric(sect[[grep(t,colnames(sect))]])
    f = sapply(f,function(x) (x+abs(x))/2)  #sets negative values to zero
    f = sort(f)
    n = length(f)
    l = c(1:n)
    ro = sum(f*l)
    r = sum(f)
    g[t-1967] = 1-2/(n-1)*(n-ro/r)  #this is the formula for GINI
  }
  m = rbind(m,g)
}
colnames(m) <- c("Industry",1969:2000)
write.csv(m,file="D:/thor's folder/grad stuff/Transpo/Transport Econ/data/BEA/clustering/G_newx.csv", row.names=FALSE)
