import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.header('Genetic Disorder Prediction')
col1,col2=st.columns(2,gap="large")
with col1:
    age=st.number_input('Age')
    gims_val=st.radio('Does mother have any genetic disorder?',['Yes','No'])
    if gims_val=='Yes':
        gims=1
    else:
        gims=0
    Inherited_from_father_val=st.radio('Does father have any genetic disorder?',['Yes','No'])
    if Inherited_from_father_val=='Yes':
        Inherited_from_father=1
    else:
        Inherited_from_father=0
    Maternal_gene_val=st.radio('Is there any genetic disorder history from mother\'s side',['Yes','No'])
    if Maternal_gene_val=='Yes':
        Maternal_gene=1
    else:
        Maternal_gene=0
    Paternal_gene_val=st.radio('Is there any genetic disorder history from father\'s side',['Yes','No'])
    if Paternal_gene_val=='Yes':
        Paternal_gene=1
    else:
        Paternal_gene=0
    Blood_cell_count=st.number_input('Blood cell count (mcL)\(4.0-6.0)')

    Mother_age=st.number_input('Mother\'s age')
    Father_age=st.number_input('Father\'s age')
    Respiratory_Rate_val=st.radio('Respiratory Rate (breaths/min)',['Normal\(30-60)','Tachypnea\(above 60)'])
    if Respiratory_Rate_val=='Tachypnea':
        Respiratory_Rate=1
    else:
        Respiratory_Rate=0
    Heart_Rate_val=st.radio('Heart Rate (beats/min)',['Normal\(30-60)','Tachycard\(above 60)'])
    if Heart_Rate_val=='Tachypnea':
        Heart_Rate=1
    else:
        Heart_Rate=0
    gender_val=st.radio('Gender',['Not known','Female','Male'])
    if gender_val=='Not known':
        gender=0
    elif gender_val=='Female':
        gender=1
    else:
        gender=2
    ba_val=st.radio('Is lack of oxygen or blood flow observed?',['No','Not Applicable','Yes'])
    if ba_val=='Yes':
        ba=2
    elif ba_val=='Not Applicable':
        ba=1
    else:
        ba=0
with col2:
    asbd_val=st.radio('Any birth defects noticed during autopsy?',['No','Not Applicable','Yes'])
    if asbd_val=='Yes':
        asbd=2
    elif asbd_val=='Not Applicable':
        asbd=1
    else:
        asbd=0


    fad_val=st.radio('Supplementation during pregnancy?',['Yes','No'])
    if fad_val=='Yes':
        fad=1
    else:
        fad=0
    hore_val=st.radio('Is there any exposure to radiation?',['Yes','No'])
    if hore_val=='Yes':
        hore=1
    else:
        hore=0
    hosa_val=st.radio('Is there any consumption of drugs?',['Yes','No'])
    if hosa_val=='Yes':
        hosa=1
    else:
        hosa=0
    ac_val=st.radio('Any treatment used for infertility?',['Yes','No'])
    if ac_val=='Yes':
        ac=1
    else:
        ac=0
    happ_val=st.radio('Any anomalies in previous pregnancies?',['Yes','No'])
    if happ_val=='Yes':
        happ=1
    else:
        happ=0

    previous_abortion=st.number_input('No. of previous abortion')
    bd_val=st.radio('Any Birth Defects Defects observed?',['Singular','Multiple','Not Applicable'])
    if bd_val=='Multiple':
        bd=0
    else:
        bd=1
    s1_val=st.radio('Weightloss',['Yes','No'])
    if s1_val=='Yes':
        s1=1
    else:
        s1=0
    s2_val=st.radio('Weak Muscles or muscle spasms',['Yes','No'])
    if s2_val=='Yes':
        s2=1
    else:
        s2=0
    s3_val=st.radio('Joint Pain',['Yes','No'])
    if s3_val=='Yes':
        s3=1
    else:
        s3=0
    s4_val=st.radio('Slow senses',['Yes','No'])
    if s4_val=='Yes':
        s4=1
    else:
        s4=0
    s5_val=st.radio('Skin dryness',['Yes','No'])
    if s5_val=='Yes':
        s5=1
    else:
        s5=0
 # Dictionary mapping disorder class labels to their names
disorder_labels = {0: "Mitochondrial genetic inheritance disorder", 1: "Multifactorial genetic inheritance disorder", 2:"Single gene inheritance diseases",3:"No disorder"}  # Update with your actual labels

# Dictionary mapping disorder subclass labels to their names
subclass_labels = {0: "Alzheimer's", 1: "Cancer", 2:"Cystic Fibrosis" ,3:"Diabetes", 4:"Hemochromatosis",5:"Lebers Hereditary Optic Neuropathy",6:"Leigh Syndrome",7:"Mitochondrial Myopathy",8:"Tay Sachs",9:"No disorder"}  # Update with your actual subclass labels

if st.button('Predict'):
    if gims==0 and Inherited_from_father==0 and Maternal_gene==0 and Paternal_gene==0 and Respiratory_Rate==0 and Heart_Rate==0 and (ba==1 or ba==0) and (asbd==1 or asbd==0) and fad==0 and hore==0 and hosa==0 and ac==0 and happ==0 and previous_abortion==0 and bd==1 and s1==0 and s2==0 and s3==0 and s4==0 and s5==0:
       st.write("Genetic Disorder")
       st.text("No disorder")
       st.write("Disorder Subclass")
       st.text("No disorder subclass") 
    else:
        result1,result2 = predict(np.array([[age,gims,Inherited_from_father ,Maternal_gene,Paternal_gene,Blood_cell_count,Mother_age,Father_age,Respiratory_Rate,Heart_Rate,gender,ba,asbd,fad,hore,hosa,ac,happ,previous_abortion,bd,s1,s2,s3,s4,s5]]))
        gd=disorder_labels[result1[0]]
        ds=subclass_labels[result2[0]]

        if ds=='No disorder':
          st.write("Genetic Disorder")
          st.text("No disorder")
          st.write("Disorder Subclass")
          st.text("No disorder subclass")   
        else:
            st.write("Genetic Disorder")
            st.text(disorder_labels[result1[0]])
            st.write("Disorder Subclass")
            st.text(subclass_labels[result2[0]])
    