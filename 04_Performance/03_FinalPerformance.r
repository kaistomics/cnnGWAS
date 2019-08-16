#!/usr/bin/R

options(stringsAsFactors=F)
args = commandArgs(trailingOnly=T)
DIR_Prep=args[1]
DIR_CNN =args[2]
fname   =args[3]
DIS     =args[4:length(args)]
library(ggplot2)
library(reshape2)
library(grid)
library(gridExtra)
library(gplots)
library(plyr)

## READ CAUSAL PROBABILITY ##
df=NULL
for (DISEASE in DIS){
 f = read.csv(pipe(paste('python FinalPerformance.py',DIR_Prep, DIR_CNN, DISEASE)),header=F,sep='')
 df= rbind(df,f)
}
colnames(df)=c('disease','sensitivity','specificity','AUC','F1')
df0= transform(df, AE=rep(c('AE','-'), length(DIS) ))

df1= melt(subset(df0, select=c('disease','AUC','AE')))
p1 = ggplot(df1, aes(x=variable, y=value*100, fill=AE))+
     geom_bar(position=position_dodge(.9),stat='identity',alpha=.6,width=.8,colour='black')+
     scale_fill_manual(values=c('grey30','red'))+
     geom_text(aes(label=round(value*100,2), x=variable, y=value*100+.05), size=3,position=position_dodge(.9),vjust=-.2)+
     geom_text(aes(label=AE, x=variable, y=85), size=3,position=position_dodge(.9))+
     theme_bw()+
     labs(x='Diseases', y='Performance')+
     ggtitle('')+
     theme(panel.grid.major.y=element_blank(),legend.position='none')+
     scale_y_continuous(breaks=seq(0.,100,5))+
     coord_cartesian(ylim=c(0, 100))+ facet_grid(.~disease )

df2= melt(subset(df0, select=c('disease','F1', 'AE')))
p2 = ggplot(df2, aes(x=variable, y=value*100, fill=AE))+
     geom_bar(position=position_dodge(.9),stat='identity',alpha=.6,width=.8,colour='black')+
     scale_fill_manual(values=c('grey30','red'))+
     geom_text(aes(label=round(value*100,2), x=variable, y=value*100), size=3,position=position_dodge(.9),vjust=-.2)+
     geom_text(aes(label=AE, x=variable, y=65), size=3,position=position_dodge(.9))+
     theme_bw()+
     labs(x='Diseases', y='Performance')+
     ggtitle('')+
     theme(panel.grid.major.y=element_blank(),legend.position='none')+
     scale_y_continuous(breaks=seq(0.,100,5))+
     coord_cartesian(ylim=c(0, 100))+ facet_grid(.~disease )

pdf(fname,width=7,height=3)
plot(p1)
plot(p2)
dev.off()
