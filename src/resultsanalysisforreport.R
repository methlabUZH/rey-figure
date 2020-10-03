rm(list=ls(all=TRUE))

bin=function(x){
  bins=c(0,(7:18)*2)
  bin=numeric(0)
  for (i in x){
  bin=c(bin,max(which(bins<=i))-1)
  }
  return(bin)
}

path='C:/Users/Christoph/Documents/ETH/git/dslab/summaries/validation_results/validation_results__fold'

data=data.frame()
for (fold in 0:9){
  data=rbind(data,read.csv(paste(path,fold,'.csv',sep="")))
}
#data from error file
data=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/summaries/test_results.csv', header=TRUE)
ind=which(data$label=="label")
data=data[-ind,]
data$label=as.numeric(as.character(data$true.label))
data$prediction=as.numeric(as.character(data$prediction))
data$abs_error=as.numeric(as.character(data$abs_error))
data$prediction[data$prediction>36]=36
data$prediction[data$prediction<0]=0
data$labelbin=bin(data$label)
data$predictionbin=bin(data$prediction)
#binned:
data$prediction=round(data$prediction)
data$label=bin(data$true.label)
data$prediction[data$prediction>12]=12
data$prediction[data$prediction<0]=0


png("C:/Users/Christoph/Documents/ETH/DSLab/ratmod.png")
plot(data$label,data$prediction,xlab="Median of online ratings", ylab="Rating from model", cex.lab=1.5)
lines(0:36,0:36,col=2,lwd=4)
dev.off()
lines(0:36,-3:33,col=1)
lines(0:36,3:39,col=1)
data$squared=(data$label-data$prediction)^2
MSEs=c(5.606853,5.652237,5.84643,5.899903,5.64106)
binMSEs=c(1.064415,1.058916,1.051846,1.057345) #1.142969 for best normal model
precisions=c(0.4909662,0.4980361,0.4988217,0.5019639) #0.4909662 for best normal model
#random number generator decide to analyse bin model 2
data$predc=apply(data[,4:13],1,var)
se=matrix(NA,nrow=1273,ncol = 10)
for (i in 1:10){
  se[,i]=(data$label-data[,i+3])^2
}

buckets <- c(0,1,2,3,4,5)
mydata_hist <- hist(sqrt(data$squared), breaks=buckets, plot=FALSE)
bp <- barplot(mydata_hist$count, log="y", col="white", names.arg=buckets)
text(bp, mydata_hist$counts, labels=mydata_hist$counts, pos=1)
png("C:/Users/Christoph/Documents/ETH/DSLab/errorhist.png")
hist(sqrt(data$squared),breaks=seq(-0.5,5.5,1),col=5,xlab="Bin error",main = "",cex.lab=1.5)
dev.off()
plot(data$label,data$squared)
human=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/new_data/MSEperfigure', header=FALSE)
colnames(human)=c('filename','error')
human$filename <- lapply(human$filename, as.character)
data$file.name <- lapply(data$filename, as.character)
n<-dim(human)[1]

ind<-numeric(0)
for (i in  1:n){
  if (human[[i,1]] %in% data$file.name){
    ind<-c(ind,i)
  }
}
human=human[ind,]
n<-dim(human)[1]
model_mse<-numeric(n)
for (i in  1:n){
  model_mse[i]=data$squared[which(data$file.name==human[[i,1]])]
  
}
png("C:/Users/Christoph/Documents/ETH/DSLab/figwisebinonline.png")

plot((human$error),(model_mse), col=3-1*(model_mse>human$error)-2*(model_mse==human$error),  xlab="Mean squared bin error of online raters", ylab="Squared bin error of model",cex.lab=1.5)
lines(0:200,0:200,lwd=2)
legend('right', legend=c("16%", "16%", "68%"),
       col=c("red", "black", "green"),pch=1, cex=0.8)
dev.off()
sum(model_mse<human$error)/n
sum(model_mse>human$error)/n
sum(model_mse==human$error)/n
cor(human$error,model_mse)
require(lattice)
qqplot(data$squared,human$error)
lines(0:round(max(human$error)),0:round(max(human$error)))
qhum=quantile(human$error,(0:100)/100)
qmod=quantile(data$squared,(0:100)/100)
plot(qmod,type='l',ylim = c(0,25),xlab="%",ylab = "Quantile",col=1)

lines(qhum,col=2)
legend('topleft', legend=c("Model", "Human"),
       col=c(1,2),lty = 1, cex=0.8)
#clinician comparison
clin1=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/new_data/brugger_trainval.csv', header=FALSE)
clin2=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/new_data/mexico_trainval.csv', header=FALSE)
clin3=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/new_data/colombia_trainval.csv', header=FALSE)
clin4=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/new_data/brugger_test.csv', header=FALSE)
clin5=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/new_data/mexico_test.csv', header=FALSE)
clin6=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/new_data/colombia_test.csv', header=FALSE)
clin=rbind(clin1,clin2,clin3,clin4,clin5,clin6)
clin$V1=as.character(clin$V1)
n<-dim(clin)[1]
ind<-numeric(0)
for (i in  1:n){
  if (clin[[i,1]] %in% data$filename){
    ind<-c(ind,i)
  }
}
clintest=clin[ind,]
n<-dim(clintest)[1]
clintest$V3=clintest$V2 # squared error
model_mse<-numeric(n)
for (i in  1:n){
  model_mse[i]=data$squared[which(data$filename==clintest[[i,1]])]
  clintest[i,3]=(clintest[i,2]-data$true.label[which(data$filename==clintest[[i,1]])])^2
  
}
ind=which(clintest$V3<100)
model_mse_rel=model_mse[ind]
clintest_rel=clintest[ind,]

#binning
clintest_rel$bin=bin(clintest_rel$V2)
n<-dim(clintest_rel)[1]
for (i in  1:n){
  clintest_rel[i,3]=(clintest_rel[i,4]-data$label[which(data$filename==clintest_rel[[i,1]])])^2
  
}
#figurewise comparison clinician vs. model
png("C:/Users/Christoph/Documents/ETH/DSLab/figwisebinclin.png")
plot(clintest_rel$V3+rnorm(n,0,0.05),model_mse_rel+rnorm(n,0,0.05), col=3-1*(model_mse_rel>clintest_rel$V3)-2*(model_mse_rel==clintest_rel$V3) , xlab="Squared bin error of clinician", ylab="Squared bin error of model",cex.lab=1.5)
lines(0:1000,0:1000,lwd=2)
legend('right', legend=c("17%", "42%", "41%"),
       col=c("red","black", "green"),pch=1, cex=0.8)
dev.off()
sum(model_mse_rel<clintest_rel$V3)/n
sum(model_mse_rel==clintest_rel$V3)/n
sum(model_mse_rel>clintest_rel$V3)/n


rater=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/new_data/dlf', header=FALSE)
rater$V1=as.character(rater$V1)


png("C:/Users/Christoph/Documents/ETH/DSLab/histsscore.png")
hist(rater$V2,breaks=seq(-0.5,36.5,1),main = "",xlab = "Median Score", ylab="Number of images",col=5,cex.lab=1.5,axes = FALSE)
axis(side=1,at=c(seq(0,36,6)))
axis(side=2)
while (!is.null(dev.list()))  dev.off()
rater$V3=bin(rater$V2)
png("C:/Users/Christoph/Documents/ETH/DSLab/histsbin.png")
hist(rater$V3,breaks=seq(-0.5,12.5,1),main = "",xlab = "Bin of Median Score", ylab="Number of images",col=5,cex.lab=1.5)
while (!is.null(dev.list()))  dev.off()
n<-dim(clin)[1]
ind<-numeric(0)
for (i in  1:n){
  if (clin[[i,1]] %in% rater$V1){
    ind<-c(ind,i)
  }
}
clinrat=clin[ind,]
n<-dim(clinrat)[1]
clinrat$V3=clinrat$V2

for (i in  1:n){
  clinrat[i,3]=rater$V2[which(rater$V1==clinrat[i,1])]
}
clinrat$V4=(clinrat$V3-clinrat$V2)^2
clinratbet=clinrat[clinrat$V4<100,]
clinrat$binclin=bin(clinrat$V2)
clinrat$binrat=bin(clinrat$V3)
clinratbet$binclin=bin(clinratbet$V2)
clinratbet$binrat=bin(clinratbet$V3)
clinratbet$binsquared=(clinratbet$binrat-clinratbet$binclin)^2
qclin=quantile(clinratbet$V4,(0:100)/100)

#qqqplot
png("C:/Users/Christoph/Documents/ETH/DSLab/qqqbin.png")

plot(qmod,type='l',ylim = c(0,25),xlab="%",ylab = "Quantile",col=1,cex.lab=1.5,lwd=2)
lines(qclin,col=3,lwd=2)
lines(qhum,col=2,lwd=2)
legend('topleft', legend=c("Model", "Raters", "Clinician"),
       col=c(1,2,3),lty = 1, cex=1)
dev.off()

par(mfrow=c(2,1))
hist(clinrat$V2,main = "Histogram of clinicians' ratings",xlab="Label",breaks=seq(0,40,5))
hist(clinrat$V3,main = "Histogram of workers' ratings",xlab="Label",breaks=seq(0,40,5))
wilcox.test(clinrat$V2,clinrat$V3,paired = TRUE,alternative = "g")

par(mfrow=c(1,1))

dev.copy(png,'myplot.png')
dev.off()
png("C:/Users/Christoph/Documents/ETH/DSLab/clinrat.png")
par(bg=NA)
plot(clinrat$V3,clinrat$V2,xlab = "Median of online ratings",ylab="Clinician's rating", cex.lab =1.5)
lines(0:36,0:36,col=2,lwd=4)
dev.off()
png("C:/Users/Christoph/Documents/ETH/DSLab/clinratbet.png")
par(bg=NA)
plot(clinratbet$V3,clinratbet$V2,xlab = "Median of online ratings",ylab="Clinician's rating", cex.lab =1.5)
lines(0:36,0:36,col=2,lwd=4)
dev.off()
qcrat=quantile(clinrat$V2,(0:100)/100)
qwrat=quantile(clinrat$V3,(0:100)/100)
plot(qwrat,qcrat,xlab = "Quantile of online ratings",ylab="Quantile of clinician's rating")
lines(0:36,0:36,col=2)



for (fold in 0:9){
  data_n=read.csv(paste(path,fold,'.csv',sep=""))
  mypath <- file.path("C:","Users/Christoph/Documents/ETH/git/dslab",paste("histfold", fold, ".jpg", sep = ""))
  jpeg(file=mypath)
  hist(data_n$label,main = paste("Histogram of Fold",fold),xlab="Label",breaks=seq(0,40,5))
  dev.off()
}
plot(data$label,data$prediction,xlab="Label",ylab="Prediction")
lines(0:36,0:36,col=2,lwd=5)
data$error=abs(data$label-data$prediction)

p=unique(data$label)
p=p[order(p)]
MSEind<-numeric(length(p))
meanErr<-numeric(length(p))
for (i in 1:length(p)){
  subdata=data[which(data$label==p[i]),]
  MSEind[i]=mean((subdata$prediction-subdata$label)^2)
  meanErr[i]=mean(subdata$error)
}
plot(p,MSEind, ylab="Mean Squared Error per Label",xlab="Label")
plot(p,meanErr,ylab="Mean Error per Label",xlab="Label")
abline(h=30,col=2)
data$file.name[order(data$error,decreasing = TRUE)[1:10]]
levels(data$data.set)[1]="Brugger_Daten"
worst<-data[order(data$error,decreasing = TRUE)[1:100],1:4]
write.table(worst, file = "C:/Users/Christoph/Documents/ETH/git/dslab/data/raw/worst.csv", sep = ",", col.names = FALSE, row.names = F,
            qmethod = "double")


value=runif(20,20,30)
dat$group="A"
dat$group[11:20]="B"
dat$group=as.factor(dat$group)



library(tidyverse)
#cnn progress
val1=c(32.90959962005961, 42.55838522877501, 31.04428297253658, 49.77641182028252, 26.682953661936477, 32.669321066474716, 31.21220908564571, 45.87954716348665, 32.196448876428455, 28.437631739294343)
val2=c(37.05147124576407, 38.92586521896857, 34.92190307932692, 42.948454183861905, 28.589644595038173, 34.619747488647896, 38.81276655259255, 38.1492679138022, 31.108576575805028, 25.982992337325477)
val3=c(30.07047975729777, 32.09226079444951, 24.62033726008707, 36.44616063845958, 23.22894653411214, 28.376085718603626, 32.57871573834343, 34.76463837444661, 23.867086962114413, 25.3382651737806)
val4=c(42.44471158738056, 31.47866429794089, 22.066591417787237, 35.527420240956346, 29.918553577913386, 22.289185587211005, 28.425665407340578, 25.926050934750325, 28.333619952524558, 22.458070561247393)
val5=c(25.26691012206162, 27.802611923965898, 22.528882351028233, 37.483570854128985, 22.11521072450696, 27.227855488410604, 28.524207837604614, 49.255897731345634, 24.90829298891086, 24.186727186437032)
val6=c(28.202136607740115, 30.27536813575615, 20.953233006695786, 46.60721395171936, 34.34779145331137, 23.700556828029523, 35.27529911189843, 25.127069338913984, 25.798425482889545, 20.66509688977557)
val7=c(27.712080107575364, 31.360124073591184, 22.918037587134467, 32.978088241699176, 28.662183732423195, 22.615246661474416, 34.38310171548281, 27.23616246168454, 21.886221855301272, 19.080923260858256)
val8=c(32.19877544315754, 35.82010976164795, 25.124320959905415, 35.36147562641551, 24.52888214371205, 31.39329179788163, 33.697365868072474, 26.516513738902123, 24.29947354779455, 23.254896655258058)
val9=c(28.528102697169285, 38.11160399172983, 30.234004766595206, 37.806861898973565, 24.621102043921812, 36.04736816179718, 35.39664295776574, 28.471803307965732, 29.366014915371693, 26.079795668140925)
val10=c(25.564128425749665, 31.420344565655615, 21.426671690828783, 30.612941584229304, 16.891503192539634, 23.337329511265924, 28.956293675075212, 29.893698557348515, 19.851858613163728, 19.57172902694305)
val11=c(25.82572366787943, 27.471217941308314, 21.93135556708665, 31.28234617913416, 16.04859363822698, 21.504897785628973, 28.560346995363602, 24.40221032604917, 19.774412285501402, 18.50520780250945)
group=factor(c("no","cd","norm+cd", "min+cd","min+norm+cd","gam+min+cd","all","all(dropout=0.85)"))
group=factor(group,levels = c("no","cd","norm+cd", "min+cd","min+norm+cd","gam+min+cd","all","all(dropout=0.85)"))
mn=c(mean(val1),mean(val2),mean(val3),mean(val4),mean(val6),mean(val7),mean(val10),mean(val11))
ci=qt(0.975,9)
sdev=c(sd(val1),sd(val2),sd(val3),sd(val4),sd(val6),sd(val7),sd(val10),sd(val11))
dat2=data.frame(Progress=group,mean=mn,sd=sdev)
ggplot(dat2)+geom_errorbar( aes(x=group, ymin=mn-ci*sd, ymax=mn+ci*sd), width=0.2, colour=2, alpha=0.9, size=1.3)+ggtitle("95% Confidence Bounds Under Normality")
ggplot(dat2)+geom_errorbar( aes(x=Progress, ymin=mean-sd, ymax=mean+sd), width=0.2, colour=2, alpha=0.9, size=1.3)+ggtitle("1 Standard Deviation Bars")+geom_point(aes(x=Progress, mean))+ylab("MSE")
size=factor(c("116*150","58*76","232*300"))
size=factor(size,levels=c("116*150","58*76","232*300"))
mn2=c(mean(val7),mean(val8),mean(val9))
sd2=c(sd(val7),sd(val8),sd(val9))
datsz=data.frame(Image_Size=size,mean=mn2,sd=sd2)
ggplot(datsz)+geom_errorbar( aes(x=Image_Size, ymin=mean-sd, ymax=mean+sd), width=0.2, colour=2, alpha=0.9, size=1.3)+ggtitle("1 Standard Deviation Bars")+geom_point(aes(x=Image_Size, mean))+ylab("MSE")+xlab("Image Shape")
wilcox.test(val10,val4,alternative = "less",paired = TRUE)
t.test(val7,val4,alternative = "less",paired = TRUE)
vals=c(val1,val2,val3,val4,val5)
groups=c(rep("no",10),rep("cd",10),rep("norm+cd",10),rep("min+cd",10),rep("norm+min+cd",10))
groups=factor(groups,levels=c("no","cd","norm+cd","min+cd","norm+min+cd"))
dat3=data.frame(group=groups,val=vals)
boxplot(val~group,data=dat3)
val.frame=data.frame(no=val1,cd=val2,normcd=val3,mincd=val4,normmincd=val5)
cor(val.frame)

#baseline
val1=c(59.705381941974316, 51.12426211167578, 51.89112989012985, 50.36828400469898, 41.184417263833566, 51.296938758786155, 63.2441533951715, 55.38893030899253, 52.12143191112637, 45.41937367499916)
val2=c(87.1039, 76.38635357142856, 65.96572916666666, 77.70665229885057, 71.31076149425286, 73.74051724137931, 93.29652298850574, 98.7296443965517, 77.09482040229886, 81.84375)
val3=c(69.52953942857143, 54.06341542857144, 53.36049022988506, 56.04673908045977, 50.85812931034482, 49.795325287356334, 58.94331091954023, 64.76120804597701, 61.955435057471284, 49.446722988505755)
val4=c(44.003802929833725, 44.0858940351918, 35.62152693989795, 54.319909839452386, 29.193215255757114, 34.97812412653054, 45.064124759100025, 47.42832826790085, 40.708547236577104, 35.00058204532033)
val5=c(25.82572366787943, 27.471217941308314, 21.93135556708665, 31.28234617913416, 16.04859363822698, 21.504897785628973, 28.560346995363602, 24.40221032604917, 19.774412285501402, 18.50520780250945)
mod=factor(c("Linear Regression", "KNN", "Random Forest", "AdaBoost", "CNN"))
mod=factor(mod,levels = c("Linear Regression", "KNN", "Random Forest", "AdaBoost", "CNN"))
mn=c(mean(val1),mean(val2),mean(val3),mean(val4),mean(val5))
ci=qt(0.975,9)
sdev=c(sd(val1),sd(val2),sd(val3),sd(val4),sd(val5))
dat2=data.frame(Model=mod,mean=mn,sd=sdev)
ggplot(dat2)+geom_errorbar( aes(x=Model, ymin=mean-sd, ymax=mean+sd), width=0.2, colour=2, alpha=0.9, size=1.3)+ggtitle("1 Standard Deviation Bars")+geom_point(aes(x=Model, mean))+ylab("MSE")


#bins
val1=c(3.3776223776223775, 2.9114219114219115, 3.51981351981352, 2.9696969696969697, 3.41025641025641, 2.9228971962616823, 3.0841121495327104, 3.5163551401869158, 3.6098130841121496, 2.9813084112149535)
val2=c(2.822044220635454, 2.453868427994104, 3.4225569482093343, 3.4408087409125527, 2.999505044146543, 2.762375009710092, 3.0555548781859514, 2.6711031886218883, 3.2178158394467333, 3.4664595629774384)
val3=c(2.8811188811188813, 3.006993006993007, 3.1818181818181817, 3.027972027972028, 3.1445221445221447, 2.7663551401869158, 2.6658878504672896, 3.1985981308411215, 3.2406542056074765, 2.824766355140187)
val4=c(4.468531468531468, 5.076923076923077, 4.941724941724941, 4.445221445221446, 5.142191142191142, 4.567757009345795, 3.8130841121495327, 4.200934579439252, 5.453271028037383, 3.7429906542056073)

mod=factor(c("MSE", "Integer Bins", "Continuous Bins", "One Hot"))
mod=factor(mod,levels = c("MSE", "Integer Bins", "Continuous Bins", "One Hot"))
mn=c(mean(val1),mean(val2),mean(val3),mean(val4))
ci=qt(0.975,9)
sdev=c(sd(val1),sd(val2),sd(val3),sd(val4))
dat2=data.frame(Model=mod,mean=mn,sd=sdev)
ggplot(dat2)+geom_errorbar( aes(x=Model, ymin=mean-sd, ymax=mean+sd), width=0.2, colour=2, alpha=0.9, size=1.3)+ggtitle("1 Standard Deviation Bars")+geom_point(aes(x=Model, mean))+ylab("Bin MSE")+xlab("Training")+geom_hline(yintercept = 2.41,col=3)

#augmented
val1=c(28.257440283970073, 26.95336878610235, 21.113089100894854, 32.15613885192919, 19.88699758934226, 23.171920333161623, 25.712813114935685, 25.78534922483298,18.558013343761562, 18.90675800106578)


#var compare
fw=read.csv('C:/Users/Christoph/Documents/ETH/git/dslab/new_data/forwilcox.csv', header=FALSE)
comp=fw$V2
inc=fw$V1[1:1918]
all=c(comp,inc)
nsim=1000000
sim_diff=numeric(nsim)
for (i in 1:nsim){
  inc_ind=sample(1:4570,1918,replace = FALSE)
  sim_inc=all[inc_ind]
  sim_comp=all[-inc_ind]
  sim_diff[i]=mean(sim_inc)-mean(sim_comp)
}
min(sim_diff)
max(sim_diff)
obs_diff=mean(comp)-mean(inc)


#convergence
library(tidyverse)
nimg=c(1000,2000,3000,4000,5000,5090)
conv1000=c(12.730974149708187,12.734588934340298,13.36272977507845,13.390928883367351,13.288610259285818,13.389899271348662)
conv2000=c(11.272981104219838,11.818106567969753,11.31307176361752,11.83815884204091,11.503995297049617,11.562631018993859)
conv3000=c(10.696351499536863,10.806285045240173,10.521295542411302,10.802162096584706,10.528174852451313,10.516713208216762)
conv4000=c(9.882823318424316,10.037784077800385,9.768777278783697,9.771634138889207,9.735055777562938,9.859330863055773)
conv5000=c(9.762387471098654,9.26595558197352,9.184129324394984,9.097953392857116,9.394469160146693,9.565808485200604)
convall=c(9.033709898343115,9.013354058825525,9.073141334757732,8.974397654783795,8.88160595305141,9.237390426602001)
MSE=c(mean(conv1000),mean(conv2000),mean(conv3000),mean(conv4000),mean(conv5000),mean(convall))
sd=c(sd(conv1000),sd(conv2000),sd(conv3000),sd(conv4000),sd(conv5000),sd(convall))
maxc=c(max(conv1000),max(conv2000),max(conv3000),max(conv4000),max(conv5000),max(convall))
minc=c(min(conv1000),min(conv2000),min(conv3000),min(conv4000),min(conv5000),min(convall))
convdat=data.frame(nimg=nimg,mean=MSE,sd=sd,max=maxc,min=minc)
png("C:/Users/Christoph/Documents/ETH/DSLab/conv.png")
ggplot(convdat[1:5,],aes(x=nimg,y=mean))+geom_errorbar( aes(ymin=mean-sd, ymax=mean+sd), width=200, colour=2, alpha=1, size=1)+geom_point(aes(x=nimg, mean))+ylab("MSE")+xlab("Number of images used")+theme_grey(base_size = 17)
dev.off()
ggplot(convdat, aes(x=nimg, y=mean, 
                       ymin=mean-sd, ymax=mean+sd))+
  geom_point(pch=1)+
  geom_errorbar(width=0.9,col=2)
+ggtitle("1 Standard Deviation Bars")
MSE=c(13.34654110015378,11.62694979409317,11.088576923023762,10.645777313817465,9.648062634772218,9.590911533733609)
plot(nimg,MSE, xlab="Number of Images Used")

binMSE=c(1.807746281223688,1.5972888015717093,1.4309233791748528,1.042829076620825)
MSE=c(9.190473534321143,8.236670876279081,8.184786252540686,5.61)
method=as.factor(c("one","one and DA","one-per-item and DA","averaged"))
method=factor(method,levels=c("one","one and DA","one-per-item and DA","averaged"))
png("C:/Users/Christoph/Documents/ETH/DSLab/binevolution.png")
plot(as.numeric(method),binMSE,type="l",xaxt = "n",lwd=4,xlab = "Training procedure", ylab = "Validation bin MSE",cex.lab=1.5)
axis(1, labels = as.character(method), at = as.numeric(method))
points(as.numeric(method),binMSE,cex=2,pch=16)
dev.off()
png("C:/Users/Christoph/Documents/ETH/DSLab/evolution.png")
plot(as.numeric(method),MSE,type="l",xaxt = "n",lwd=4,xlab = "Training procedure", ylab = "Validation MSE",cex.lab=1.5)
axis(1, labels = as.character(method), at = as.numeric(method))
points(as.numeric(method),MSE,cex=2,pch=16)
dev.off()

