#!/usr/bin/R
library(plyr)
###############################################
options(stringsAsFactors=F)
args   =commandArgs(trailingOnly =T)
Disease=args[1]             # Disease="AUTsummary_hg19"
F_raw  =args[2]             # F_raw="../00_RawDatas/AUTsummary_hg19.txt"
DIR_imp=args[3]             # DIR_imp="../01_RunImpG/Data/"
DIR_out=args[4]             # DIR_out="./SigBlock" 
CUTOFF =as.numeric(args[5]) # CUTOFF=as.numeric(5e-04)
###############################################

 f.write_input = file(paste(DIR_out,'/',Disease,'.input.txt',  sep=''),"w")
 f.write_bed   = file(paste(DIR_out,'/',Disease,'.sig.bed',    sep=''),"w")
 f.write_spec  = file(paste(DIR_out,'/',Disease,'.Range_spec' ,sep=''),"w")

 ##################
 # READ GWAS FILE
 RawFile = read.table(F_raw,header=T)
 colnames(RawFile) = c("snpid","chr","bp","a1","a2","zscore")
 RawFile = transform(RawFile, pval= 2*pnorm(-abs(zscore)))

 for (i in 1:22) {
  ####################
  # SUBSET RAW FILE
  chrom = paste('chr',i,sep=''); print(chrom)
  RawFile_filt = subset(RawFile, chr==chrom & pval< CUTOFF)
  if (nrow(RawFile_filt)==0 ) next

  ####################
  # READ IMPUTED FILE
  Imputed =  read.table(paste(DIR_imp,"/",Disease,"/all.",chrom,".impz.bed",sep=''),header=F)
  colnames(Imputed)=c('chrom','start','bp','snpid','a1','a2','zscore','r2pred')
  Imputed =  transform(Imputed, pval= 2*pnorm(-abs(zscore))) # calculate p-value
  Imputed_filt = subset(Imputed, pval < CUTOFF)                  # filtering
  ######################
  # MERGE RAW & IMPUTED
  merged_= rbind( subset(RawFile_filt, select=c('bp','a1','a2','pval')), 
                  subset(Imputed_filt, select=c('bp','a1','a2','pval')))
  merged= ddply(merged_, c('bp','a1','a2'), summarise, pval=min(pval))
  merged= merged[order(merged$pval),] # sorting

  #################
  # GET TAG SNP
  Interval = 1000000 # 1Mb range Interval between lead SNPs 
  merged   = transform(merged, START= ifelse(bp-Interval <0, 0, bp- Interval), 
                               END  = bp+Interval)  
  LeadSNPs = merged[1,]
  for ( j in 2:nrow(merged) ){
    testingSNP_bp = merged[j,'bp']
    if (sum(LeadSNPs$START <= testingSNP_bp &
            LeadSNPs$END   >= testingSNP_bp ) > 0) {
      next
    } else{
     LeadSNPs = rbind(LeadSNPs, merged[j,])
    }}
  LeadSNPs = LeadSNPs[order(LeadSNPs$bp),]

  ##################
  # Function
  getBlock = function(bp, intv, chrom=NULL, saveFile=NULL) {
   start= ifelse(bp-intv<0, 0, bp-intv)
   end  = bp+intv
   if (!is.null(saveFile)){
     writeLines( paste(chrom, start, end,sep='\t'), saveFile)
    }
   return(list( start, end))
  }
  ##################
  # Write Files
  for ( j in 1:nrow(LeadSNPs) ){
   leadSNP_bp   = LeadSNPs[j,'bp']
   leadSNP_pval = LeadSNPs[j,'pval']
   Block_out = getBlock( leadSNP_bp, 500000)

   testingBlock  = subset(merged, Block_out[[1]] <= bp &
                                  Block_out[[2]] >= bp )

   if ( nrow(testingBlock) < 3 ) next

   end_idx= ifelse(nrow(testingBlock)>30, 30, nrow(testingBlock))
   leadSNPID = paste(chrom,':',leadSNP_bp,sep='')
   writeLines( paste(chrom,
                     leadSNPID, 
                     paste(testingBlock$bp[1:end_idx],
                     collapse=','), sep=','),f.write_input)
   writeLines( paste(chrom, 
                     min(testingBlock$bp[1:end_idx]), 
                     max(testingBlock$bp[1:end_idx]),
                     leadSNPID ,sep='\t'),   f.write_spec)

   for (m in 1:end_idx){
    writeLines(paste(chrom, 
                     testingBlock[m,'bp']-1, 
                     testingBlock[m,'bp'],
                     leadSNPID,
                     testingBlock[m,'a1'],
                     testingBlock[m,'a2'],
                     Block_out[[1]], Block_out[[2]], 
                     testingBlock[m,'pval'],
                     sep="\t"), f.write_bed)
  }}}
 close(f.write_input)
 close(f.write_bed)
 close(f.write_spec)

