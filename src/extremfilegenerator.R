rm(list=ls(all=TRUE))
#path where all folds results are stored + beginning of filename
path='/Users/jheitz/git/dslab/new_data/val_predictions/20181110_14-15-08_new_data3__fold'

data=data.frame()
for (fold in 0:9){
  data=rbind(data,read.csv(paste(path,fold,'.csv',sep="")))
}
data$error=abs(data$label-data$prediction)
levels(data$data.set)[1]="Brugger_Daten"
worst<-data[order(data$error,decreasing = TRUE)[1:100],1:4]

#store csv to desired destination with desired name
write.table(worst, file = "/Users/jheitz/git/dslab/new_data/val_predictions/20181110_14-15-08_new_data3__worst.csv", sep = ",", col.names = FALSE, row.names = F,
            qmethod = "double")