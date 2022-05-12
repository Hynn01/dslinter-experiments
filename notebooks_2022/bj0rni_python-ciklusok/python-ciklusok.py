#!/usr/bin/env python
# coding: utf-8

# # Python ciklusok
# 
# A ciklus eredetileg valamilyen ismétlődő dolgot jelent, gondolhatunk holdciklusra, választási ciklusra, árapályciklusra. Mi ebben a könyvben egy olyan programrészletet értünk rajta, amely valahányszor megismétlődik. Futtassuk le a következő példát és próbáljuk értelmezni a kódot:

# In[ ]:


számláló = 1
while számláló < 10:
    print(számláló)
    számláló = számláló + 1
print("Viszlát.")


# A program sorról sorra:
# - `számláló = 1` létrehoz egy változót,
# - `while számláló < 10:` a beljebb kezdett programrész mindaddig fog futni, amíg a `while` utáni rész igaz, azaz a `számláló` értéke kisebb $10$-nél (ez nagyon hasonlít az előző órán vett `if`-es dolgokhoz),
# - `print(számláló)` minden körben kiírja a `számláló` értékét, majd
# - `számláló = számláló + 1` megnöveli $1$-gyel.
# 
# A program tehát annyit tesz, hogy mindaddig írogatja ki a `számláló` értékét, amíg az $10$ alatt van, és mindig megnöveli $1$-gyel. Amint a `számláló` eléri a $10$-et, immár nem lesz igaz a `while` utáni feltétel, és a ciklus befejeződik.
# 
# **FONTOS!** Ne felejtsük ki a `számláló = számláló + 1` sort; enélkül a feltétel mindig igaz lesz és a programunk *végtelen ciklusba* kerül. Ebben az esetben a bal oldali gombbal gyorsan lődd le a programot, mielőtt kifagyasztja a böngészőt.

# **1. feladat (Büntetőfogalmazás).** Írjuk le százszor, hogy „Nem pofázok az informatikaórán”! Az előző kódon mindössze egy sort kell módosítani.

# In[ ]:





# **2. feladat.** Írjuk ki minden harmadik számot $1$-tól $100$-ig! ($1, 4, 7, 11, \ldots$)

# In[ ]:





# **3. feladat.** Írjuk ki az első tíz négyzetszámot! ($1, 4, 9, 16, \ldots$)

# In[ ]:





# ## Túltenni Gauss-on
# 
# A kis Gauss, amikor még nem volt nagy matematikus, az anekdota szerint egész osztályával
# együtt azt a feladatot kapta a tanárától, hogy adja össze a számokat egytől százig. A tanár
# közben nekiállt valami más munkának, de a kis Gauss két perc múlva szólt, hogy készen
# van, és az eredmény $5050$. Gauss egy ügyes trükköt használt, de ha már van számítógépünk, illendő a két percnél jobb eredményt hoznunk,
# méghozzá Gauss felismerésének kihasználása nélkül. Valahogy így (próbáld meg értelmezni a kódot!):

# In[ ]:


számláló = 1
összeg = 0
while számláló <= 100:
    összeg = összeg + számláló
    számláló = számláló + 1
print('Összesen:', összeg)


# A program mindössze annyit tesz, hogy ugyanúgy elszámol $1$-től $100$-ig a `számláló` változóval, de előtte létrehoz egy `összeg` változót is, amelynek kezdetben $0$ az értéke. Minden körben megnöveli az `összeg`-et a `számláló` aktuális értékével, azaz tényleg az $1+2+3+4+\ldots+99+100$ számot számolja ki.
# 
# **4. feladat.** Számoljuk ki a $20!$ értékét, azaz szorozzuk össze a számokat $1$-től $20$-ig (ciklussal)! Segítség: hozzunk létre egy `szorzat` változót a **megfelelő kezdőértékkel**, és minden körben szorozzuk hozzá a `számláló` változót.

# In[ ]:





# ## Egyenletmegoldás a csöves-módszerrel
# 
# Ciklussal meg tudunk oldani egyenleteket az egész számok halmazán, a jó öreg próbálgatós módszerrel, kihasználva, hogy a Python sokkal gyorsabban próbálkozik, mint mi. Oldjuk meg például a 3x + 2 = 59 egyenletet a pozitív egészek halmazán (a `!=` szimbólumot még nem láttuk, ez csupán a $\neq$ jele Python-ban):

# In[ ]:


x = 1
while 3*x + 2 != 59:
    x = x + 1
print('A megoldás:', x)


# A program addig próbálkozik egyre növekvő $x$-ekkel, amíg nem teljesül az egyenlőség. Amint teljesül, a ciklus kilép, és a program kiírja a megoldást.

# **5. feladat.** Oldjuk meg a $6x^2 + 3x + 8 = 767$ egyenletet az egész számok halmazán!

# In[ ]:




