#!/usr/bin/R
require(plyr)
require(ggplot2)

options(width=200,stringsAsFactors=F)
args = commandArgs(trailingOnly=T)
DIR  = args[1]
fname= args[2]
disease=args[3:length(args)]

df =NULL
for (i in 1:length(disease)){
 f  = read.csv(pipe(paste('python getAUC.py ',DIR,'/',disease[i],'_out',sep="")),sep='\t')

 ff = transform(f, model= paste(AE,sep='_'))
 ff = ddply(ff, 'model', transform,  best_F1=max(F1))
 ff_= subset(ff, F1==best_F1)
 ff_= ff_[order(-ff_$AUC),]

  f11 = with(subset(ff_, AE=='False')[1,],data.frame(disease=disease[i], AE='',   score=F1, group='F1' ))
  f12 = with(subset(ff_, AE=='False')[1,],data.frame(disease=disease[i], AE='',   score=AUC,group='AUC'))
  f21 = with(subset(ff_, AE=='True' )[1,],data.frame(disease=disease[i], AE='AE', score=F1, group='F1' ))
  f22 = with(subset(ff_, AE=='True' )[1,],data.frame(disease=disease[i], AE='AE', score=AUC,group='AUC'))

 df = rbind(df, rbind(f11,f12,f21,f22))
}
 df = transform(df, group=factor(group, levels=c('F1','AUC')))

 p1 = ggplot(df, aes(x=group, y=score, fill=AE))+
     geom_bar(position=position_dodge(.9),stat='identity',alpha=.6,width=.8,colour='black')+
     scale_fill_manual(values=c('grey20','red'))+
     geom_text(aes(label=score, x=group, y=score), size=3,position=position_dodge(.9),vjust=-.2)+
     geom_text(aes(label=AE, x=group, y=5), size=3,position=position_dodge(.9))+
     theme_bw()+
     labs(x='Diseases', y='Performance')+
     ggtitle('Using Auto-encoder')+
     theme(panel.grid.major.y=element_blank(),legend.position='none')+
     scale_y_continuous(breaks=seq(0.,100,10))+
     coord_cartesian(ylim=c(0, 100))+ facet_grid(.~disease )

 pdf(fname,width=10,height=3)
 plot(p1)
 dev.off()


