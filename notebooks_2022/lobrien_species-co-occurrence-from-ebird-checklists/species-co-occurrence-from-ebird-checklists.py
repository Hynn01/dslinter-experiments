#!/usr/bin/env python
# coding: utf-8

# # Species Co-Occurrence Based on eBird Checklists
# <a id="intro"></a>
# 
# This notebook is intended for support of [BirdCLEF 2022](https://www.kaggle.com/competitions/birdclef-2022). It's laughably late in the competition, but maybe it'll help someone or future students. The output of this notebook is a co-occurrence matrix of bird species at various birding hotspots in a variety of habitats. The hypothesis is that distinct habitats will have different species present: you won't find a shorebird such as a Dunlin (`dunlin`) in the upland forests where you might find (if you're lucky) an iʻiwi (`iiwi`).
# 
# [![birdlcef.jpg](https://i.postimg.cc/xTpGV2qW/birdlcef.jpg)](https://postimg.cc/gnRL31VD)
# 
# <a id="get-data"></a>
# ## Getting and using eBird data
# 
# The data in this notebook come from eBird, are owned or licensed by the Cornell Lab of Ornithology, and are shared in accordance with [eBird's terms and conditions](https://www.birds.cornell.edu/home/terms-of-use/) and the rules of Kaggle's BirdCLEF 2022. Editorial comments are my own and should not be lent any credence.
# 
# To run this notebook, you will need an eBird API key, which you can get by choosing ['Request access' from the **API Key** row here](https://ebird.org/data/download). Once you have an API key, in this notebook select the **Add-ons** menu, **Secrets** item. Create a new secret with the name `ebird_api_key` and set the value to your API key. 
# 
# [![Screen-Shot-2022-05-07-at-10-12-08-AM.png](https://i.postimg.cc/wTRM38wK/Screen-Shot-2022-05-07-at-10-12-08-AM.png)](https://postimg.cc/561bkGcp)
# 
# <a id="missing-data"></a>
# ## A quick note about missing species
# 
# A third of species present in the training data for BirdCLEF 2022 are not in the co-occurrence matrix. The easiest interpretation is that the birds are uncommon enough that no one has reported sightings of them, but that doesn't mean they don't occur. There was an `incter1` at South Point this Winter that every bird photographer on the Big Island mobbed, but no eBird records. Or, in the case of the Puaohi `puaioh`, at least, eBird will not share known locations, since doing so may trigger an influx of potentially-disruptive viewers. 
# 
# I have made some notes about the species that are missing. Pure speculation on my part. 
# 
# <a id="olelo"></a>
# ## What's with the "apostrophes" and "long vowels"? 
# 
# ʻōlelo Hawaiʻi has relatively more vowels and diphthongs than consonants compared to English. Sometimes vowels are drawn out a little longer, which is indicated with a kahakō -- the macron over the vowel. That's pretty hard to hear, but the ʻokina ("apostrophe") is very easy to hear: it's a glottal stop, like when you say the word "uh-oh" and there's a clean break between the "uh" and the "oh". So "Hawaiʻi" is not "hah-why-yee", it's "hah-why-ee" (well actually, it's more like "ha-vye-ee" but that's kind of pretentious. Only someone who still uses "data are" would try to pull that off... *ahem*... Anyway, moving on...). 

# <a id="init"></a>
# ## Initialization

# In[ ]:


get_ipython().system('python -m pip install ebird-api')


# In[ ]:


from enum import Enum
from dataclasses import dataclass
from typing import Dict

from tqdm import tqdm 
import time

import pandas as pd
import numpy as np
import itertools
from kaggle_secrets import UserSecretsClient

import seaborn as sns # Easy-to-use heatmap plot

from ebird.api import Client


# In[ ]:


# Species abbreviations in Bird CLEF contest are 1:1 correspondence to eBird field codes for that species. So these are all valid values in `myRecord.speciesCode)`
species = 'afrsil1 akekee akepa1 akiapo akikik amewig aniani apapan arcter                       barpet bcnher belkin1 bkbplo bknsti bkwpet blkfra blknod bongul                       brant brnboo brnnod brnowl brtcur bubsan buffle bulpet burpar buwtea                       cacgoo1 calqua cangoo canvas caster1 categr chbsan chemun chukar cintea                       comgal1 commyn compea comsan comwax coopet crehon dunlin elepai ercfra eurwig                       fragul gadwal gamqua glwgul gnwtea golphe grbher3 grefri gresca gryfra gwfgoo                       hawama hawcoo hawcre hawgoo hawhaw hawpet1 hoomer houfin houspa hudgod iiwi incter1                       jabwar japqua kalphe kauama laugul layalb lcspet leasan leater1 lessca lesyel lobdow lotjae                       madpet magpet1 mallar3 masboo mauala maupar merlin mitpar moudov norcar norhar2 normoc norpin                       norsho nutman oahama omao osprey pagplo palila parjae pecsan peflov perfal pibgre pomjae puaioh                       reccar redava redjun redpha1 refboo rempar rettro ribgul rinduc rinphe rocpig rorpar rudtur ruff                       saffin sander semplo sheowl shtsan skylar snogoo sooshe sooter1 sopsku1 sora spodov sposan                       towsol wantat1 warwhe1 wesmea wessan wetshe whfibi whiter whttro wiltur yebcar yefcan zebdov'.split()


# In[ ]:


# This is my internal model of likely soundscapes in Hawai'i

class Habitat(Enum):
    UplandForest = 0,
    LowlandForest = 1,
    UplandDry = 2,
    Grasslands = 3,
    Cloudforest = 4,
    Swamp = 5,
    Human = 6,
    Shoreline = 7,
    Pelagic = 8,
    Freshwater = 9


# In[ ]:


@dataclass
class Hotspot:
    name : str
    habitat : Habitat
    location_code : str


# In[ ]:


# These are some notable hotspots, mostly on Hawai'i Island, but Hosmer's on Maui and Alakai's on Kauai.
# These might be helpful for characterizing a soundscape, given a signature species 

hotspots = [
    Hotspot('keahole', Habitat.Shoreline, 'L2574849'),
    Hotspot('kaloko', Habitat.Shoreline, 'L331009'),
    Hotspot('pu`u la`au', Habitat.UplandDry, 'L285813'),
    Hotspot('VNP Thurston', Habitat.LowlandForest, 'L837288'),
    Hotspot('Hakalau', Habitat.UplandForest, 'L695639'),
    Hotspot('Hosmer Grove', Habitat.UplandForest, 'L247163'),
    Hotspot('Kaloka Mauka', Habitat.Cloudforest, 'L8599253'),
    Hotspot('Kailua Kona', Habitat.Human, 'L577366'),
    Hotspot('Waiki`i', Habitat.Grasslands, 'L1430539'),
    Hotspot('Alakai Swamp Trail', Habitat.Swamp, 'L1548049'),
    Hotspot('Honokohau Offshote', Habitat.Pelagic, 'L1662616'),
    Hotspot('Wailoa River', Habitat.Freshwater, 'L868863')
]


# In[ ]:


#To run this notebook, you will need an eBird API key, which you can get by choosing 'Request access' from the **API Key** row at 
# https://ebird.org/data/download 
# Once you have an API key, in this notebook select the **Add-ons** menu, **Secrets** item. 
# Create a new secret with the name `ebird_api_key` and set the value to your API key. 

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("ebird_api_key")


# In[ ]:


locale = 'en'
client = Client(api_key, locale)


# In[ ]:


FETCH_FROM_EBIRD = True # Or False to re-use data from an earlier run
if FETCH_FROM_EBIRD :
    df = pd.DataFrame()
    for s in tqdm(species):
        rs = client.get_species_observations(s, 'US-HI')
        for r in rs:
            record_df  = pd.DataFrame([r], columns=r.keys())
            df = pd.concat([df, record_df], axis =0).reset_index(drop=True)
    filtered = df[df.howMany > 0]
    filtered.to_csv("ebird_recs.csv", index = False)
else:
    filtered = pd.read_csv("../input/ebird_recs.csv")


# <a id="ebird-recs"></a>
# ## Take a look at the eBird records
# 
# Just a quick look at what you get from eBird... 

# In[ ]:


filtered[filtered.locId == 'L695639']


# <a id="missing-species"></a>
# ## Missing species
# 
# Not all species in training data have records in Hawaiʻi in eBird.

# In[ ]:


found_species = filtered.speciesCode.unique()
missing_species = set(species) - set(found_species)
print(sorted(missing_species))


# <a id="missing-notes"></a>
# ## Notes on missing species
# 
# - `akekee` : Akekee are critically-endangered, single habitat is above mosquito line in Kauai
# - `akikik` : Akikiki are critically-endangered, single habitat is above mosquito line in Kauai
# - `amewig` : American Wigeon. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `belkin1` : Belted Kingfisher. I've never seen a Kingfisher in Hawaiʻi. Presumably co-occurs with Wailoa River (L868863)
# - `bongul` : Bonaparte's Gull. Like all gulls but Laughing Gull uncommon-rare in Hawaiʻi.
# - `bubsan`: Buff-breasted sandpiper. Cute as anything. Shoreline habitat. 
# - `buffle` : Bufflehead. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `buwtea` : Blue-winged teal. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `cangoo` : Canadian goose. I suppose the key co-occur for this would be Nene (`hawgoo`)
# - `canvas`: Canvasback. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `caster1`: Caspian tern. I've never seen one in Hawaiʻi. Pelagic. 
# - `cintea` : Cinnamon teal. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863).
# - `comsan`: Common sandpiper. Shoreline habitat. 
# - `crehon` : Akohekohe. Upland forest habitat.  Found only on Maui.
# - `eurwig` : European wigeon. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `fragul` : Franklin's gull. Like all gulls but Laughing Gull uncommon-rare in Hawaiʻi.
# - `gadwal` : Gadwall. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `gamqua` : Gambell's Quail. Desert bird, so Upland Dry, I suppose, possible co-occur with Palila
# - `glwgul` : Glaucous gull. Like all gulls but Laughing Gull uncommon-rare in Hawaiʻi.
# - `gnwtea` : Green-winged teal.  Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `golphe` : Golden pheasant. Wow, I didn't know we had these. Uhh... I guess anywhere you'd get Kalij Pheasant (`kalphe`)? 
# - `grbher3` : Great blue heron. Never seen one in Hawaiʻi. Probably anywhere you'd get a Night Heron (`bcnher`)
# - `gresca` : Greater scaup.  Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `gwfgoo` : Greater white-fronted goose. I suppose the key co-occur for this would be Nene (`hawgoo`)
# - `hoomer` : Hooder merganser. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `hudgod` : Hudsonian godwit. Shoreline habitat. Kaloko Fishpond (L331009).
# - `incter1` : Inca tern. Rare enough so that every birder and photographer on Hawaiʻi Island mobbed the single individual at South Point this winter. I'm pretty sure this ain't gonna' be in the test-set. (But, uh, Shoreline or Pelagic habitat.)
# - `japqua` : Japanese quail. Grassland habitat, maybe Upland Dry. 
# - `leasan` : Least sandpiper. Shoreline habitat. 
# - `lessca` : Lesser scaup. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `lesyel` : Lesser yellowlegs. Shoreline habitat.
# - `lobdow` : Long-billed dowitcher. Shoreline habitat. 
# - `madpet` : Zino's petrel. Pelagic habitat. 
# - `magpet1` : Magenta petrel. Critically endangered. Found only in South Pacific. Pelagic habitat. Virtually zero chance in test set. 
# - `maupar` : Maui parrotbill. Critically endangered. Upland Forest. Found only in Maui. 
# - `merlin` : Merlin. Fast raptor common on mainland US. There was one not far from L285813 about 5 years ago. So Upland Dry, but I doubt it's in the test set.
# - `mitpar` : Mitred parakeet. Co-occurs with `rempar`. Maybe uncommon, but definitely not rare. Keep this one in your training set for Human habitats.
# - `norhar2` : Northern Harrier. Never seen one in Hawaii, presumably co-occurs would be anywhere you have Iʻo (`hawhaw`)
# - `norpin` : Northern pintail. Duck, so presumably co-occurs from Kaloko Fishpond (L331009) or Wailoa River (L868863)
# - `osprey` : Osprey. Never seen one in Hawaii. Pelagic, Riverine, Shoreline habitats. If I "heard an Osprey", I'd guess `hawhaw`. 
# - `parjae` : Parasitic jaeger. Pelagic habitat. Seen 'em, never heard one. 
# - `pecsan` : Pectoral sandpiper. Shoreline habitat. 
# - `puaioh` : Puaiohi are critically-endangered, single habitat is above mosquito line in Kauai. (On eBird's **sensitive species** list, so will not show in checklists.)
# - `redpha1` : Red phalarope. Seasonally common in Pelagic habitat. 
# - `ribgul` : Ring-billed gull. Like all gulls but Laughing Gull uncommon-rare in Hawaiʻi.
# - `ruff` : Ruff. Shoreline habitat, presumably co-occurs would be from Kaloko Fishpond (L331009). Never seen one, but I think they may only be "uncommon" and not all-out "rare". 
# - `shtsan` : Sharp-tailed sandpiper.  Shoreline habitat. 
# - `sopsku1`: South-polar skua. Pelagic habitat. 
# - `sora` : Sora are small rails. Presumably co-occurs would be Wailoa River (L868863).
# - `sposan` : Spotted sandpiper. Shoreline habitat. 
# - `towsol` : Towsend's solitaire. Never heard of it. I guess it would be Upland or Lowland Forest and maybe Human habitats (?). 
# - `wessan` : Western sandpiper. Shoreline habitat.
# 
# ## More opinions on uncommon / rare species
# 
# Iʻo (`hawhaw`, "Hawaiian Hawk") are locally common -- I see them every day. If you get a hit on a hawk call (`"cheer"`), it's _almost certainly_ a `hawhaw`.
# 
# There are a lot of locally-uncommon ducks. These might be confounders or it might be the case that there's a "duck pond" hotspot in the test set. 
# 
# Ditto sandpipers. I expect there are some shoreline soundscapes in the test, but I sincerely doubt they've caught some of these rarities in a natural setting.
# 
# As for the rare pelagics: No way. Unless they cut and paste a recording of a Magenta Petrel into a pelagic soundscape, it's gonna' be a bad guess. I spend a lot of time offshore and 99% of the birds you see you don't hear. If there's a sea-based soundscape in the test set, I'd guess it's either going to be just off a breeding site for tropic birds (`whttro` and `redtro`) or noddies (`blknod`, `brwnod`). 
# 
# Pelagics in general: My guess is that the test set might well have land-based breeding and nesting calls of locally-rare pelagic birds (albatrosses, for instance, and very possibly the endangered petrel ʻuaʻu `hawpet1`). There's been a big effort to create safe breeding areas to reintroduce some of these birds to the main islands and finding and monitoring nests is a big challenge.

# <a id="cooccur"></a>
# ## Co-occurrence

# In[ ]:


co_occurs = {} # Dict<SpeciesCode> -> Dict<SpeciesCode> -> TimesPresent (not sum(HowMany))  
locs = filtered.groupby(filtered.locId)
for _, loc_records_df in locs:
    species_here = loc_records_df.speciesCode.unique()
    location_co_occurs = list(itertools.product(species_here, species_here))   
    records_here_by_species = loc_records_df.groupby(loc_records_df.speciesCode)
    for k1 in species_here:
        if k1 not in co_occurs : 
            co_occurs[k1] = {}
        k1_co_occurs = co_occurs[k1]
        for k2 in species_here:
            if k2 not in k1_co_occurs : 
                k1_co_occurs[k2] = 0
            k1_co_occurs[k2] += 1 # Presence noted 

# Convert to Pandas and sort both rows and columns by field code
df = pd.DataFrame(co_occurs).sort_index()
df = df.reindex(sorted(df.columns), axis=1)
df.to_csv("ebird_co_occurrances_absolute.csv")


# In[ ]:


co_occur_prob_df = df.copy()
for species in co_occur_prob_df.columns:
    times_seen = co_occur_prob_df[species][species] # Co-occurs w. self == count
    co_occur_prob_df[species] = co_occur_prob_df[species] / times_seen
co_occur_prob_df.to_csv("ebird_co_occurrances_probabilities.csv")
co_occur_prob_df


# In[ ]:


sns.set(rc={'figure.figsize':(30,30)})
sns.heatmap(co_occur_prob_df, cmap=sns.color_palette("viridis_r", as_cmap=True))


# In[ ]:


Probability = float # Range: 0.0 - 1.0
Species = str # Field code

def probability_of_s2_given_s1(prob_df : pd.DataFrame, s1 : Species, s2 : species) -> Probability : 
    return prob_df[s1][s2]

# Apapane are widespread upland forest birds, I'iwi are in fewer locations
common_given_rare = probability_of_s2_given_s1(co_occur_prob_df, 'iiwi', 'apapan')
rare_given_common = probability_of_s2_given_s1(co_occur_prob_df, 'apapan', 'iiwi')
# Probs should be ~1.0 of Apapane (widespread) if I'iwi (sparser)
(common_given_rare, rare_given_common)


# In[ ]:


def co_occurrence_probabilities(prob_df : pd.DataFrame, given: Species) -> Dict[Species, Probability] : 
    return prob_df[given].dropna().to_dict()
   
co_occurrence_probabilities(co_occur_prob_df, 'iiwi')


# In[ ]:


len(co_occurrence_probabilities(co_occur_prob_df, 'zebdov')), len(co_occurrence_probabilities(co_occur_prob_df, 'iiwi'))


# <a id="cooccur-notes"></a>
# ## Notes
# 
# This looks pretty reasonable to me: 
# 
# - widespread commoners like mynahs, house finches and sparrows, doves, and white-eyes. 
# - birder target species like i'iwi have reasonable co-occurs although biased towards birder interest: you don't truly have a 25% chance of a pueo (`sheowl`) given an i'iwi (`iiwi`) in any soundscape I know of. But in a couple hours / single checklist I can likely bag a (rare) `sheowl` and (rare) `iiwi`. 

# <a id="sentinels"></a>
# ## Signature species for habitats / soundscapes
# 
# TODO: What birds are most characteristic of a particular habitat? 
# 
# * Expand `hotspots` list
# * Use `client.get_observations(locationCode)` for h in hotspots
# * Add `habitat` col to obs
# * You want the most common bird seen in `h` that is least likely to occur in `!h`
# * This might be more effort than worth, since observed != heard. You have to use domain knowledge, i.e., tropic birds are noisier than noddies. 
# 
# 

# <a id="scorebirds"></a>
# ## Only scoring birds

# In[ ]:


scored_birds = ["akiapo", "aniani", "apapan", "barpet", "crehon", "elepai", "ercfra", "hawama", "hawcre", "hawgoo", "hawhaw", "hawpet1", "houfin", "iiwi", "jabwar", "maupar", "omao", "puaioh", "skylar", "warwhe1", "yefcan"]
unscored_birds = set(found_species) - set(scored_birds)
print(sorted(unscored_birds))


# They aren't even _trying_ to find palila? 😢
# 
# ...
# 
# Rows (note transpose in spec) are species that is "given". Color is probability of col species.  

# In[ ]:


sns.set(rc={'figure.figsize':(30,4)})
xdf = co_occur_prob_df.copy()
#xdf = xdf.drop(index = unscored_birds)
xdf = xdf.drop(columns = unscored_birds)
sns.heatmap(xdf.T, cmap=sns.color_palette("viridis_r", as_cmap=True))


# In[ ]:


sns.set(rc={'figure.figsize':(8,8)})
xdf = xdf.drop(index = unscored_birds)
sns.heatmap(xdf.T, cmap=sns.color_palette("viridis_r", as_cmap=True))


# 
