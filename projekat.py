import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb
from scipy.stats import f_oneway

# ucitavanje podataka
data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

#pregled ucitanih podataka
print(data.head())
print(data.info())
columns = data.columns
print(columns)

#ciscenje podataka; uklanjanje duplikata i nedostajucih vrednosti
data = data.drop_duplicates()

#osnosvna statistika ociscenih podataka (count, mean, std, min, 25%, 50%, 75%, max)
basic_stats = data.describe()
print(basic_stats)

#sve vrednosti koje se javljaju u razlicitim kategorijama dataset-a
column1 = 'Gender'
all1 = data[column1].unique()
print("Pol sve vrednosti: ", all1)

column2 = 'Occupation'
all2 = data[column2].unique()
print("Zanimanja: ", all2)

column3 = 'Quality of Sleep'
all3 = data[column3].unique()
print("Kvalitet sna sve vrednosti: ", all3)

column4 = 'BMI Category'
all4 = data[column4].unique()
print("BMI sve vrednosti: ", all4)

column5 = 'Sleep Disorder'
all5 = data[column5].unique()
print("Poremecaj sna sve vrednosti: ", all5)

column6= 'Age'
all6 = data[column6].unique()
print("Godine sve vrednosti: ", all6)

column7= 'Stress Level'
all7 = data[column7].unique()
print("Nivo stresa sve vrednosti: ", all7)

column8= 'Physical Activity Level'
all8 = data[column8].unique()
print("Nivo fizicke aktivnosti sve vrednosti: ", all8)

column9= 'Sleep Duration'
all9 = data[column9].unique()
print("Duzina sna sve vrednosti: ", all9)

#klasifikacija podataka i prikaz
sb.boxplot(x='Gender', y='Quality of Sleep', data=data)
plt.xlabel('Pol')
plt.ylabel('Kvalitet sna')
plt.title('Kvalitet sna prema polu')
plt.show()

#anova za polove, da li je razlika u kvalitetu sna kod polova statisticki znacajna 
#filtriranje podataka po polu koje je neophodno za ANOVA analizu
male_data = data[data['Gender'] == 'Male']['Quality of Sleep']
female_data = data[data['Gender'] == 'Female']['Quality of Sleep']

#ANOVA analiza (oneway)
anova, p_value = f_oneway(male_data, female_data)

#ispis rezultata
print('Anova: ', anova)
print('p vrednost: ', p_value)

#korelacija izmedju godina i kvaliteta sna 
age = data['Age'] # x-osa
quality_of_sleep = data['Quality of Sleep'] # y-osa
fit = np.polyfit(age, quality_of_sleep, 1)
fit_fn = np.poly1d(fit)

korelacija = np.corrcoef(age, quality_of_sleep)[0, 1]
print("Korelacija godina i kvaliteta sna: ", korelacija)

sb.scatterplot(x='Age', y='Quality of Sleep', data=data)
plt.plot(age, fit_fn(age), color='red')
plt.xlabel('Godine')
plt.ylabel('Kvalitet sna')
plt.title('Kvalitet sna prema godinama')
plt.show()

#korelacija nivoa stresa i kvaliteta sna i scatter plot u kome su tacke obojene po zanimanjima
stress = data['Stress Level']
quality_of_sleep = data['Quality of Sleep']
fit1 = np.polyfit(stress, quality_of_sleep, 1)
fit_fn1 = np.poly1d(fit1)

korelacija1 = np.corrcoef(stress, quality_of_sleep)[0, 1]
print("Korelacija nivoa stresa i kvaliteta sna: ", korelacija1)

color = sb.color_palette('muted', len(data['Occupation'].unique()))

sb.scatterplot(data=data, x='Stress Level', y='Quality of Sleep', hue='Occupation', palette=color, s=100)
plt.plot(stress, fit_fn1(stress), color='red')
plt.xlabel('Nivo stresa')
plt.ylabel('Kvalitet sna')
plt.title('Correlation between Stress Level and Quality of Sleep')
plt.legend(title='Occupation')
plt.tight_layout()
plt.show()

#boxplot za BMI i kvalitet sna; ANOVA analiza za proveru da li je razlika u kalitetu sna za razlicite kategorije BMI-ja statisticki znacajna
sb.boxplot(x='BMI Category', y='Quality of Sleep', data=data)
plt.xlabel('BMI')
plt.ylabel('Kvalitet sna')
plt.title('Kvalitet sna prema BMI kategoriji')
plt.show()

#ANOVA analiza izmedju svake BMI kategorije kako bih odredila da li je statisticki znacajna razlika kod kvaliteta sna
bmi_categories = ['Normal', 'Normal Weight', 'Overweight', 'Obese']
results = pd.DataFrame(index=bmi_categories, columns=bmi_categories, dtype=str)

for category1 in bmi_categories:
    for category2 in bmi_categories:
        if category1 == category2:
            results.loc[category1, category2] = '-'
        else:
            data_category1 = data[data['BMI Category'] == category1]['Quality of Sleep']
            data_category2 = data[data['BMI Category'] == category2]['Quality of Sleep']

            anova_result = f_oneway(data_category1, data_category2)

            nivo_znacajnosti = 0.05
            if anova_result.pvalue < nivo_znacajnosti:
                results.loc[category1, category2] = '✓'
            else:
                results.loc[category1, category2] = '✗'

results = results.rename_axis("Statisticki znacajna razlika kvaliteta sna", axis="columns")
print(results)

#histogram za duzinu spavanja sa oznacnim disorderimma 
insomnia_data = data[data['Sleep Disorder'] == 'Insomnia']['Sleep Duration']
sleep_apnea_data = data[data['Sleep Disorder'] == 'Sleep Apnea']['Sleep Duration']

plt.hist([insomnia_data, sleep_apnea_data], bins=10, color=['blue', 'orange'], label=['Insomnia', 'Sleep Apnea'])
plt.xlabel('Duzina sna')
plt.ylabel('Frequency')
plt.title('Histogram koji prikazuje duzinu sna sa obelezenim poremecajima sna')
plt.legend()
plt.show()

#korelacija izmedju fizicke aktivnosti i kvaliteta sna 
#scatter plot
physical_activity = data['Physical Activity Level'] # x-osa
quality_of_sleep = data['Quality of Sleep'] # y-osa
fit2 = np.polyfit(physical_activity, quality_of_sleep, 1)
fit_fn2 = np.poly1d(fit2)

korelacija = np.corrcoef(physical_activity, quality_of_sleep)[0, 1]
print("Korelacija fizicke aktivnosti i kvaliteta sna: ", korelacija)

sb.scatterplot(x='Physical Activity Level', y='Quality of Sleep', data=data)
plt.plot(physical_activity, fit_fn2(physical_activity), color='red')
plt.xlabel('Fizicka aktivnost')
plt.ylabel('Kvalitet sna')
plt.title('Kvalitet sna prema fizickoj aktivnosti')
plt.show()

#korelacija izmedju fizicke aktivnosti i duzine sna 
#scatter plot
physical_activity = data['Physical Activity Level'] # x-osa
sleep_duration = data['Sleep Duration'] # y-osa
fit3 = np.polyfit(physical_activity, sleep_duration, 1)
fit_fn3 = np.poly1d(fit3)

korelacija = np.corrcoef(physical_activity, sleep_duration)[0, 1]
print("Korelacija fizicke aktivnosti i duzine sna: ", korelacija)

sb.scatterplot(x='Physical Activity Level', y='Sleep Duration', data=data)
plt.plot(physical_activity, fit_fn3(physical_activity), color='red')
plt.xlabel('Fizicka aktivnost')
plt.ylabel('Duzina sna')
plt.title('Duzina sna prema fizickoj aktivnosti')
plt.show()

#otkucaji srca i kvalitet sna
heart_rate = data['Heart Rate'] # x-osa
sleep_quality= data['Quality of Sleep'] # y-osa
fit4 = np.polyfit(heart_rate, quality_of_sleep, 1)
fit_fn4 = np.poly1d(fit4)

korelacija = np.corrcoef(heart_rate, sleep_quality)[0, 1]
print("Korelacija kvaliteta sna i otkucaja srca: ", korelacija)

sb.scatterplot(x='Heart Rate', y='Quality of Sleep', data=data)
plt.plot(heart_rate, fit_fn4(heart_rate), color='red')
plt.xlabel('Otkucaji srca')
plt.ylabel('Kvalitet sna ')
plt.title('Kvalitet sna u zavisnosti od otkucaja srca')
plt.show()

