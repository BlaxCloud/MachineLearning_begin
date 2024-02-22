# Listing 4.26
# Это промежуточный код, он не является рабочим
i1 = [-1.5, -0.75]
# i1 = [0.25, 1.1]
R1 = aln.predict(i1)
print('R1=', R1)

if (R1 == 1):
    print('R1= Сорт Iris-setosa')
else:
    print('R1= Сорт Iris-versicolor')
