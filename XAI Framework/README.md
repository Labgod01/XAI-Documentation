# XAI Framework

Πλαίσιο εκτέλεσης μεθόδων επεξηγήσιμης τεχνητής νοημοσύνης (XAI) πάνω σε μοντέλα
sklearn και PyTorch. Υποστηρίζει 9 μεθόδους, για classification και regression,
με τοπικές (local) και καθολικές (global) επεξηγήσεις. Υποστηρίζει επίσης μοντέλα
multi-class, multi-label, multi-output και multi-target regression
(βλ. [Πολλαπλοί στόχοι](#πολλαπλοί-στόχοι-multi-class-multi-label-multi-output-multi-target)).

## Μέθοδοι

| Μέθοδος | Κλειδί στο `main.py` | Σημειώσεις |
|---|---|---|
| LIME | `lime` | Local explanations |
| TreeSHAP | `tree_shap` | Local και global explanations· μόνο μοντέλα δέντρων (παραλείπεται για PyTorch) |
| KernelSHAP | `kernel_shap` | Local και global explanations· το global κομμάτι μπορεί να είναι αργό |
| Counterfactuals (DiCE) | `dice` | Local explanations |
| Anchors | `anchors` | Μόνο classification |
| ALE | `ale` | Global explanations |
| Permutation Feature Importance | `pfi` | Global explanations |
| PDP / ICE | `pdp_ice` | PDP: global, ICE: local |
| Integrated Gradients | `integrated_gradients` | Local attributions και global feature importance· σε μοντέλα PyTorch εξηγεί απευθείας το δίκτυο, αλλιώς μπορεί να χρησιμοποιεί surrogate |

## Απαιτήσεις

- **Python 3.11.** Το project δοκιμάστηκε με αυτή την έκδοση. Με άλλες εκδόσεις
  δεν είναι εγγυημένο ότι θα εγκατασταθούν όλα τα πακέτα (ειδικά τα `alibi` και `dice_ml`).
- Σύνδεση στο internet για την εγκατάσταση των πακέτων.
- Περίπου 3 GB ελεύθερου χώρου (το `torch` και οι εξαρτήσεις του είναι μεγάλα).
- **Μην αναβαθμίσεις το `numpy` σε 2.x.** Το `alibi 0.9.6` (ALE) χρησιμοποιεί ένα όρισμα
  της `np.percentile` που αφαιρέθηκε στο numpy 2, και το ALE αποτυγχάνει.

## Εγκατάσταση

Από τον φάκελο του project:

**Windows (PowerShell)**

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
venv\Scripts\activate
pip install -r requirements.txt
```

Η εντολή `Set-ExecutionPolicy` χρειάζεται μόνο αν το PowerShell μπλοκάρει το
`activate`. Ισχύει μόνο για το συγκεκριμένο παράθυρο. Μετά το `activate` πρέπει
να εμφανιστεί `(venv)` στην αρχή του prompt.

**Mac / Linux**

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Η εγκατάσταση παίρνει αρκετή ώρα: κατεβαίνει το `torch` (~124 MB) και το `pip`
δοκιμάζει πολλές εκδόσεις των `spacy`/`thinc` λόγω των περιορισμών του `alibi`.
Αυτό είναι φυσιολογικό και δεν σημαίνει ότι κόλλησε.

Το `scikit-learn` είναι κλειδωμένο στην έκδοση **1.8.0**, γιατί το αρχείο
`models/my_model.joblib` αποθηκεύτηκε με αυτή. Με άλλη έκδοση εμφανίζονται
προειδοποιήσεις και η εκτέλεση γίνεται πολύ πιο αργή.

## Εκτέλεση των ελέγχων

```
python test_cases.py
```

Τρέχει και τις 9 μεθόδους στα 8 βασικά cases και τυπώνει ένα **SUMMARY** στο τέλος.
Η εκτέλεση παίρνει αρκετά λεπτά, κυρίως λόγω του KernelSHAP.

Γρήγορος έλεγχος με δύο cases:

```
python test_cases.py --case custom_classification custom_nn_classification
```

Άλλες χρήσιμες μορφές:

```
python test_cases.py anchors                      # μία μέθοδος, όλα τα cases
python test_cases.py treeshap kernelshap          # πολλές μέθοδοι
python test_cases.py anchors --case demo_regression custom_nn_classification
python test_cases.py anchors --scope local        # local | global | both
```

### Εκτέλεση μίας μεθόδου, case και scope

```
python test_cases.py <μέθοδος> --case <case> --scope <local|global|both>
```

Παράδειγμα:

```
python test_cases.py lime --case custom_classification --scope local
```

Το αποτέλεσμα εμφανίζεται στην οθόνη (τιμές της επεξήγησης και SUMMARY) και
γράφεται στο `outputs_test/<case>/`.

### Cases

Τα 8 cases είναι συνδυασμοί: `demo` / `custom` × `classification` / `regression`
× sklearn / PyTorch NN (π.χ. `demo_regression`, `custom_nn_classification`).
Στα cases με PyTorch NN το δίκτυο τυλίγεται από το `torch_wrapper.py` και τρέχουν όλες οι
μέθοδοι (βλ. [Μοντέλα PyTorch](#μοντέλα-pytorch)).

Επιπλέον υπάρχουν 7 cases για [πολλαπλούς στόχους](#πολλαπλοί-στόχοι-multi-class-multi-label-multi-output-multi-target):

| Case | Πρόβλημα | Μοντέλο |
|---|---|---|
| `custom_multiclass` | 1 στήλη, 3 κλάσεις | `RandomForestClassifier` |
| `custom_nn_multiclass` | 1 στήλη, 3 κλάσεις | PyTorch, 3 έξοδοι (logits) |
| `custom_multilabel` | 3 στήλες 0/1 | `RandomForestClassifier` (native multi-output) |
| `custom_nn_multilabel` | 3 στήλες 0/1 | PyTorch, 3 έξοδοι (logits, `BCEWithLogitsLoss`) |
| `custom_multioutput_classification` | 1 στήλη 0/1 + 1 στήλη με 3 κλάσεις | `MultiOutputClassifier(RandomForestClassifier)` |
| `custom_multitarget_regression` | 3 αριθμητικές στήλες | `RandomForestRegressor` (native multi-output) |
| `custom_nn_multitarget_regression` | 2 αριθμητικές στήλες | PyTorch, 2 έξοδοι |

```
python test_cases.py --multi        # μόνο τα 7 multi cases
python test_cases.py --all          # και τα 15 cases
python test_cases.py lime --case custom_multilabel
```

Χωρίς `--multi` / `--all` / `--case` τρέχουν μόνο τα 8 βασικά cases, όπως πριν.
Στα cases με πολλές στήλες στόχου κάθε μέθοδος τρέχει μία φορά ανά στήλη, και στο
SUMMARY κάθε γραμμή έχει το όνομα του target, π.χ. `LIME [label_b]`.

### Τι σημαίνει το SUMMARY

| Ένδειξη | Σημασία |
|---|---|
| `OK` | Η μέθοδος ολοκληρώθηκε και παρήγαγε αρχεία. |
| `OK (no files - skipped?)` | Η μέθοδος παραλείφθηκε σκόπιμα. Συμβαίνει στα **Anchors** σε regression (υποστηρίζονται μόνο για classification) και στο **TreeSHAP** στα cases με PyTorch NN (εξηγεί μόνο μοντέλα δέντρων). Δεν είναι σφάλμα. |
| `ERROR` | Πραγματικό σφάλμα. Δίπλα φαίνεται το μήνυμα και η γραμμή του κώδικα. |

Αναμενόμενο αποτέλεσμα: καμία γραμμή `ERROR`.

## Αποτελέσματα

Τα αρχεία γράφονται στο `outputs_test/<case>/`:

- `plots/` — εικόνες (png)
- `raw_results/` — τα ακατέργαστα αποτελέσματα (json)

**Προσοχή:** κάθε εκτέλεση του `test_cases.py` σβήνει και ξαναδημιουργεί τον φάκελο
κάθε case που τρέχει. Αν τρέξεις πρώτα μία μέθοδο και μετά άλλη στο ίδιο case,
τα αρχεία της πρώτης χάνονται. Για να κρατήσεις αποτελέσματα πολλών μεθόδων,
τρέξε τις μαζί σε μία εντολή.

Ο φάκελος `outputs/` (κανονική εκτέλεση του project) δεν επηρεάζεται από τα τεστ.

## Κανονική εκτέλεση

Η εκτέλεση γίνεται μέσω του `main.py`. Όλα τα ορίσματα και οι default τιμές τους φαίνονται με:

```powershell
python main.py --help
```

### Υποχρεωτικά ορίσματα

```powershell
python main.py --data test_dataset.csv --model test_model.joblib --target price --task regression
```

| Όρισμα | Τι ορίζει |
|---|---|
| `--data` | Το CSV με τα features και τη στήλη (ή τις στήλες) στόχου |
| `--model` | Το εκπαιδευμένο μοντέλο (`.joblib`, `.pkl`, `.pt`) |
| `--target` | Η στήλη στόχου. Για μοντέλα με πολλούς στόχους, όλες οι στήλες με τη σειρά των εξόδων του μοντέλου |
| `--task` | `classification` ή `regression` |

Και τα 4 είναι **υποχρεωτικά** και δεν διαβάζονται από το `config.py`. Αν λείπει κάποιο, η εκτέλεση
σταματά με μήνυμα που λέει ποιο λείπει, π.χ.
`main.py: error: the following arguments are required: --model, --task`.

### Γραμμές χωρίς πραγματική τιμή στόχου

Η στήλη στόχου πρέπει να **υπάρχει** στο CSV (ώστε ένα λάθος στο `--target` να πιάνεται), αλλά
μπορεί να έχει **κενά κελιά**, ή να είναι ολόκληρη κενή. Αυτό χρειάζεται όταν η πραγματική τιμή
δεν έχει μετρηθεί ακόμα, π.χ. η πρόβλεψη φορτίου για αύριο.

- Τα κενά κελιά παίρνουν την **πρόβλεψη του μοντέλου**, οπότε όλες οι μέθοδοι τρέχουν κανονικά:
  ούτως ή άλλως εξηγούν τις προβλέψεις του μοντέλου, όχι τις πραγματικές τιμές.
- Το **PFI** μετράει πόσο πέφτει η απόδοση του μοντέλου, άρα χρειάζεται πραγματικές τιμές:
  χρησιμοποιεί **μόνο** τις γραμμές που τις έχουν, και παραλείπεται αν δεν υπάρχει καμία.
  Το πλήθος των γραμμών γράφεται στο `raw_results/pfi_<target>.json` (`n_rows`).
- Για κάθε στόχο με κενά τυπώνεται πόσες γραμμές συμπληρώθηκαν.
- Οι τιμές που υπάρχουν στο CSV αρχείο στη/στις στήλη/-ες στόχου θεωρούνται οι πραγματικές τιμές, διότι δεν υπάρχει τρόπος επαλήθευσης.
Αν δεν είναι (π.χ. είναι προβλέψεις του μοντέλου), το PFI θα βγάλει παραπλανητικά αποτελέσματα, γιατί δεν θα μετράει την πραγματική απόδοση του μοντέλου.

Η συμπλήρωση γίνεται στο `framework_adapter.fill_unknown_targets()`. Ο έλεγχος:
`python test_unknown_targets.py`.

### Προαιρετικά ορίσματα

Αν δεν δοθούν, παίρνουν την τιμή τους από το `config.py`. Στο `config.py` είναι σημειωμένα με `[CLI]`.

| Όρισμα | Ρύθμιση στο `config.py` | Default | Τι ορίζει |
|---|---|---|---|
| `--methods` | `xai_methods` | όλες | Ποιες XAI μέθοδοι τρέχουν |
| `--scope` | `explanation_scope` | `both` | `local`, `global` ή `both` |
| `--query-index` | `query_index` | `0` | Ποια γραμμή του dataset εξηγούν οι local μέθοδοι |
| `--class-to-explain` | `class_to_explain` | όλες οι κλάσεις (binary: η θετική) | Ποιες κλάσεις εξηγούνται: `predicted`, μια κλάση ή `στόχος=κλάση` (βλ. «Ποια κλάση εξηγείται») |
| `--output-dir` | `output_dir` | `outputs` | Ο φάκελος όπου γράφονται τα `plots/` και `raw_results/` |
| `--random-state` | `random_state` | `42` | Seed για αναπαραγώγιμα αποτελέσματα |

Ένα όρισμα αλλάζει τη ρύθμιση **μόνο για εκείνη την εκτέλεση**. Για να αλλάξει μόνιμα, άλλαξέ τη στο `config.py`.

Οι διαθέσιμες μέθοδοι για το `--methods` είναι: `lime`, `tree_shap`, `kernel_shap`, `dice`,
`anchors`, `ale`, `pfi`, `pdp_ice`, `integrated_gradients`.

```powershell
# μόνο LIME και TreeSHAP, local εξήγηση της 6ης γραμμής
python main.py --data test_dataset.csv --model test_model.joblib --target price --task regression --methods lime tree_shap --scope local --query-index 5

# εξήγηση της κλάσης 0, αποτελέσματα σε άλλο φάκελο
python main.py --data data.csv --model model.joblib --target label --task classification --class-to-explain 0 --output-dir outputs_class0
```

### Ρυθμίσεις μόνο στο `config.py`

Οι υπόλοιπες ρυθμίσεις έχουν ήδη default τιμές στο `config.py`. Τις αλλάζεις εκεί μόνο αν
χρειάζεται: συνήθως αφορούν ένα συγκεκριμένο dataset ή μοντέλο και ορίζονται μία φορά.

| Ρύθμιση | Default | Τι κάνει |
|---|---|---|
| `class_names` | από το μοντέλο | Ονόματα κλάσεων (LIME, Anchors, ALE, DiCE) |
| `immutable_features` | `[]` | Features που το DiCE δεν αλλάζει ποτέ, π.χ. `["age", "sex"]` |
| `features_to_plot` | `None` | Features στα plots του PDP/ICE και του ALE. `None`: τα 6 σημαντικότερα (PDP/ICE), τα 3 πρώτα (ALE) |
| `categorical_number_features` | `[]` | Κατηγορίες γραμμένες ως αριθμοί, π.χ. `country = 0/1/2` (LIME, Anchors, DiCE). Οι στήλες κειμένου βρίσκονται αυτόματα |
| `scoring_metric` | `accuracy` / `r2` | Μετρική του PFI, π.χ. `"accuracy"`, `"r2"` |
| `feature_metadata` | `{}` | Περιγραφές και μονάδες των features για ένα LLM. Σώζεται στο `raw_results/feature_metadata.json` |
| `ig_*` | | Ρυθμίσεις του Integrated Gradients (βήματα, baseline, surrogate) |
| `torch_*` | | Πώς διαβάζεται η έξοδος ενός δικτύου PyTorch |

Παράδειγμα `feature_metadata`:

```python
feature_metadata = {
    "age":    {"description": "Ηλικία του πελάτη", "unit": "years"},
    "income": {"description": "Ετήσιο εισόδημα", "unit": "EUR"},
}
```

### Demo

Για να δοκιμάσεις το framework χωρίς δικό σου dataset και μοντέλο:

```powershell
python main.py --demo classification       # RandomForest, classification
python main.py --demo regression           # RandomForest, regression
python main.py --demo nn_classification    # νευρωνικό δίκτυο PyTorch
python main.py --demo nn_regression
```

Με `--demo` δεν δίνονται τα `--data`, `--model`, `--target` και `--task`. Τα προαιρετικά ορίσματα ισχύουν κανονικά.

### Αποτελέσματα

Τα αποτελέσματα αποθηκεύονται στον φάκελο του `--output-dir` (default `outputs/`), στους υποφακέλους
`plots/` και `raw_results/`. Σε κάθε εκτέλεση του `main.py` σβήνονται **μόνο** αυτοί οι δύο
υποφάκελοι. Άλλα αρχεία στον ίδιο φάκελο δεν πειράζονται. Για να κρατήσεις τα αποτελέσματα
μιας εκτέλεσης, δώσε στην επόμενη άλλο `--output-dir`.

Στο τέλος της εκτέλεσης εμφανίζονται:
- Το task και το scope.
- Οι επιλεγμένες XAI μέθοδοι.
- Η διαδρομή του φακέλου αποτελεσμάτων.
- Ο αριθμός των explanations που επέστρεψε κάθε μέθοδος.

**Σημείωση:** Δεν υποστηρίζουν όλες οι μέθοδοι κάθε συνδυασμό task και scope. Για παράδειγμα, τα Anchors υποστηρίζονται μόνο για classification, ενώ τα ALE και PFI παράγουν global εξηγήσεις. Οι περιορισμοί αυτοί μπορεί να οδηγήσουν σε παράλειψη συγκεκριμένης εξήγησης χωρίς να σημαίνει απαραίτητα ότι απέτυχε ολόκληρη η εκτέλεση.

## Πολλαπλοί στόχοι (multi-class, multi-label, multi-output, multi-target)

| Πρόβλημα | Στήλες στόχου | `--task` | Παράδειγμα μοντέλου |
|---|---|---|---|
| **Multi-class** | 1 στήλη, 3+ κλάσεις | `classification` | `RandomForestClassifier`, δίκτυο με k εξόδους |
| **Multi-label** | πολλές στήλες 0/1 | `classification` | `RandomForestClassifier`, `MultiOutputClassifier`, `MLPClassifier`, δίκτυο με μία έξοδο ανά ετικέτα |
| **Multi-output classification** | πολλές στήλες, κάποιες με 3+ κλάσεις | `classification` | `RandomForestClassifier`, `MultiOutputClassifier` |
| **Multi-target regression** | πολλές αριθμητικές στήλες | `regression` | `RandomForestRegressor`, `MultiOutputRegressor`, δίκτυο με μία έξοδο ανά στόχο |

Ο τύπος προβλήματος αναγνωρίζεται αυτόματα από τον αριθμό των στηλών στόχου και τις
τιμές τους, και τυπώνεται στην αρχή της εκτέλεσης (π.χ. `Task: classification (multi-label classification)`).

### Εκτέλεση

Δίνεις όλες τις στήλες στόχου, με τη σειρά των εξόδων του μοντέλου:

```powershell
python main.py --data data/labels.csv --model models/multilabel.joblib --target label_a label_b label_c --task classification
python main.py --data data/prices.csv --model models/multireg.joblib --target y1 y2 --task regression
```


### Πώς λειτουργεί

Οι μέθοδοι XAI εξηγούν μία έξοδο τη φορά. Για μοντέλο με πολλούς στόχους, **κάθε μέθοδος
τρέχει μία φορά ανά στήλη στόχου**. Για κάθε στήλη, το `multi_output.py` δίνει στις μεθόδους
ένα «view» του μοντέλου, που επιστρέφει μόνο τα `predict()` / `predict_proba()` / `classes_`
αυτής της στήλης. Έτσι οι μέθοδοι βλέπουν ένα συνηθισμένο μοντέλο με έναν στόχο.

Τα αρχεία κάθε στόχου έχουν το όνομά του στο τέλος, π.χ. `lime_local_label_a.png`,
`pfi_y2.json`, και αποθηκεύονται όλα στον ίδιο φάκελο.

Πριν τρέξουν οι μέθοδοι, ελέγχεται ότι **ο αριθμός εξόδων του μοντέλου ταιριάζει με τον
αριθμό των στηλών στόχου**. Αν δώσεις μία στήλη σε μοντέλο με 3 εξόδους (ή το αντίθετο),
η εκτέλεση σταματά με μήνυμα, αντί να δώσει λάθος αποτελέσματα χωρίς σφάλμα.

### Ποια κλάση εξηγείται

Ορίζεται με το `--class-to-explain` ή το `class_to_explain` στο `config.py`:

| Τιμή | Τι εξηγείται |
|---|---|
| τίποτα (default) | **Όλες οι κλάσεις, μία-μία.** Στα binary μόνο η θετική κλάση (η δεύτερη του `model.classes_`), γιατί η άλλη είναι ο καθρέφτης της: ίδιες τιμές με αντίθετο πρόσημο |
| `predicted` | Μόνο η κλάση που προβλέφθηκε για τη γραμμή του `--query-index` |
| μια κλάση, π.χ. `2` | Μόνο αυτή. Με πολλούς στόχους εφαρμόζεται στους στόχους που έχουν αυτή την κλάση, οι υπόλοιποι παίρνουν το default (τυπώνεται μήνυμα). Αν κανένας στόχος δεν την έχει, η εκτέλεση σταματά με σφάλμα |
| ανά στόχο, π.χ. `risk=2 flag=predicted` | Μία ρύθμιση για κάθε στόχο. Στόχοι που δεν αναφέρονται παίρνουν το default |

```powershell
python main.py ... --class-to-explain predicted
python main.py ... --target flag risk --class-to-explain risk=2 flag=predicted
```

Στο `config.py` το ίδιο γράφεται `class_to_explain = {"risk": 2, "flag": "predicted"}`.

Όταν εξηγούνται πολλές κλάσεις, κάθε αρχείο έχει την κλάση στο τέλος, π.χ.
`lime_species_class0.json`, `lime_species_class1.json`, `lime_species_class2.json`.
Όταν εξηγείται μία, το όνομα μένει όπως πριν (`lime_species.json`). Κάθε JSON γράφει την
κλάση στο πεδίο `explained_class`.

| Μέθοδος | Τι κάνει με τις κλάσεις |
|---|---|
| LIME, TreeSHAP, KernelSHAP, PDP/ICE, Integrated Gradients | Μία εξήγηση ανά κλάση |
| DiCE | Counterfactuals **προς** κάθε κλάση, δηλαδή τι πρέπει να αλλάξει για να γίνει η πρόβλεψη αυτή η κλάση. Η κλάση που ήδη προβλέφθηκε αντικαθίσταται από όλες τις άλλες (binary: η αντίθετη) |
| Anchors | Πάντα η πρόβλεψη |
| ALE | Όλες οι κλάσεις σε ένα αποτέλεσμα |
| PFI | Δεν αφορά κλάση |

Τα `class_names` μπορούν επίσης να δοθούν ανά στόχο: `class_names = {"risk": ["low", "mid", "high"]}`.

### Περιορισμοί

- **TreeSHAP σε ένα classifier που εκπαιδεύτηκε σε πολλές στήλες μαζί** (π.χ.
  `RandomForestClassifier().fit(X, Y)` με `Y` πολλών στηλών): παραλείπεται με μήνυμα, γιατί το SHAP
  δεν δίνει σωστές τιμές ανά στόχο σε αυτή την περίπτωση. Τύλιξε το μοντέλο σε
  `MultiOutputClassifier` (ένα δέντρο ανά στόχο), ή χρησιμοποίησε KernelSHAP.
  Σε multi-target regression με `RandomForestRegressor` το TreeSHAP λειτουργεί κανονικά.
- **Chains** (`ClassifierChain`, `RegressorChain`): οι μέθοδοι τρέχουν, αλλά το TreeSHAP
  παραλείπεται.
- **Μικτοί στόχοι** (μερικές στήλες classification και μερικές regression στο ίδιο μοντέλο)
  δεν υποστηρίζονται.
- Ο χρόνος εκτέλεσης πολλαπλασιάζεται με τον αριθμό των στόχων, κυρίως στο KernelSHAP.


## Αναπαραγωγιμότητα

- Το `random_state` ορίζεται κεντρικά και περνάει σε όλες τις μεθόδους.
- Τα **Anchors** χρησιμοποιούν το ίδιο seed, οπότε δύο εκτελέσεις δίνουν τον ίδιο κανόνα.

## Μοντέλα PyTorch

Ένα δίκτυο PyTorch (`nn.Module`) δεν έχει τις εντολές `predict()`, `predict_proba()`
και `classes_`, που χρησιμοποιούν οι μέθοδοι. Τις προσθέτει το αρχείο `torch_wrapper.py`.
Το `framework_adapter.py` τυλίγει αυτόματα κάθε μοντέλο PyTorch, οπότε τρέχουν όλες οι
μέθοδοι. Τα μοντέλα sklearn δεν επηρεάζονται.

### Αποθήκευση του μοντέλου

Αν το δίκτυο εκπαιδεύτηκε σε κανονικοποιημένα δεδομένα, αποθήκευσέ το **μαζί με το scaler**:

```python
import joblib
joblib.dump({"model": net, "scaler": scaler}, "models/my_nn.joblib")
```

Το scaler ενσωματώνεται μπροστά από το δίκτυο. Οι μέθοδοι δίνουν στο μοντέλο τις αρχικές
τιμές των features, και τα attributions του Integrated Gradients βγαίνουν στις αρχικές
μονάδες. Χωρίς το scaler, το δίκτυο θα έπαιρνε μη κανονικοποιημένες τιμές και οι
προβλέψεις θα ήταν λάθος, χωρίς κανένα μήνυμα σφάλματος.

Μορφές που δέχεται το `load_model`:

| Μορφή | Πότε |
|---|---|
| `net` | Το δίκτυο δεν χρειάζεται scaler. |
| `(net, scaler)` | Δίκτυο και scaler. |
| `{"model": net, "scaler": scaler, "classes": [...], "output_type": "..."}` | Δίκτυο και scaler, με προαιρετικές ρυθμίσεις (βλ. παρακάτω). |
| αρχείο `.pt` / `.pth` από `torch.save(net, path)` | Ολόκληρο δίκτυο αποθηκευμένο με PyTorch. |

Scalers που υποστηρίζονται: `StandardScaler`, `MinMaxScaler` ή dict `{"mean": ..., "std": ...}`.
Για άλλη προεπεξεργασία, κάν' την μέσα στο `forward()` του δικτύου.

Ένα `state_dict` (μόνο τα βάρη) **δεν** γίνεται δεκτό: χρειάζεται ολόκληρο το δίκτυο.

Εκτέλεση:

```powershell
python main.py --data data/my_data.csv --model models/my_nn.joblib --target target --task classification
```

### Ρυθμίσεις στο `config.py`

| Ρύθμιση | Προεπιλογή | Πότε την αλλάζεις |
|---|---|---|
| `torch_output_type` | `"auto"` | Τι επιστρέφει το δίκτυο: `"logits"`, `"probabilities"` ή `"log_probabilities"`. Το `"auto"` κοιτάει το τελευταίο layer (`Sigmoid`/`Softmax` σημαίνει πιθανότητες, αλλιώς logits). Βάλε `"probabilities"` αν το δίκτυο κάνει `torch.sigmoid(...)` ή `torch.softmax(...)` μέσα στο `forward()`. Σε αυτή την περίπτωση εμφανίζεται και σχετική προειδοποίηση. |
| `torch_class_labels` | `None` | Οι ετικέτες των κλάσεων, με τη σειρά των εξόδων του δικτύου, π.χ. `["no", "yes"]`. Με `None` είναι `0, 1, 2, ...`. Πρέπει να ταιριάζουν με τις τιμές του target στο dataset. |

Αν το μοντέλο αποθηκεύτηκε ως dict με δικά του `"classes"` ή `"output_type"`, αυτά
υπερισχύουν του `config.py`.

### Διαφορές ανά μέθοδο

- **TreeSHAP**: παραλείπεται με μήνυμα, γιατί εξηγεί μόνο μοντέλα δέντρων. Για νευρωνικά
  χρησιμοποίησε KernelSHAP ή Integrated Gradients.
- **Integrated Gradients**: εξηγεί απευθείας το ίδιο δίκτυο με τις υπόλοιπες μεθόδους, χωρίς
  surrogate. Σε δίκτυα με μεγάλες εξόδους το convergence delta μπορεί να βγει μεγαλύτερο σε
  απόλυτη τιμή. Είναι σφάλμα ολοκλήρωσης και μικραίνει αν ανεβάσεις το `ig_n_steps`
  (π.χ. 500).
- **PFI**: τρέχει σε μία διεργασία (`n_jobs=1`), γιατί το PyTorch χρησιμοποιεί ήδη πολλούς
  πυρήνες.
- Οι υπόλοιπες μέθοδοι καλούν το δίκτυο μέσω `predict()` / `predict_proba()`, όπως ένα
  μοντέλο sklearn.

### Περιορισμοί

- Τα features πρέπει να είναι αριθμητικά. Κωδικοποίησε τα κατηγορικά πριν την εκπαίδευση.
- Αν το δίκτυο είναι δική σου κλάση (`class MyNet(nn.Module)`), το αρχείο που την ορίζει
  πρέπει να μπορεί να γίνει import όταν φορτώνεται το μοντέλο. Με `nn.Sequential` δεν
  χρειάζεται τίποτα.
- Υποστηρίζονται δίκτυα **multi-class** (μία στήλη, k έξοδοι), **multi-label** (πολλές
  στήλες 0/1, μία έξοδος ανά στήλη, logits ή sigmoid) και **multi-target regression** (μία
  έξοδος ανά στόχο). Οι έξοδοι πρέπει να έχουν τη σειρά των στηλών στο `--target`.
  Το `torch_class_labels` σε multi-label είναι οι 2 ετικέτες κάθε στήλης, π.χ. `["no", "yes"]`.
  Βλ. [Πολλαπλοί στόχοι](#πολλαπλοί-στόχοι-multi-class-multi-label-multi-output-multi-target).
- **Multi-output classification με 3+ κλάσεις σε κάποια στήλη δεν υποστηρίζεται για δίκτυα**:
  από τις εξόδους δεν φαίνεται ποιες ανήκουν σε ποια στήλη. Αν ο αριθμός εξόδων δεν ταιριάζει
  με τον αριθμό των στηλών, η εκτέλεση σταματά με μήνυμα.
- Το `joblib.load` και το `torch.load` μπορούν να εκτελέσουν κώδικα από το αρχείο.
  Φόρτωνε μόνο μοντέλα που εμπιστεύεσαι.

## Συχνά προβλήματα

- **Το PowerShell μπλοκάρει το `activate`** ("running scripts is disabled"):
  τρέξε `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` και ξαναδοκίμασε.
- **Η έξοδος γεμίζει προειδοποιήσεις του sklearn** (`delayed ... Parallel`) και η εκτέλεση
  είναι πολύ αργή: έλεγξε ότι το `scikit-learn` είναι 1.8.0 (`python -c "import sklearn; print(sklearn.__version__)"`).
- **Θέλεις log αρχείο:** χρησιμοποίησε το `cmd` για να μη χαθεί η έξοδος αν διακοπεί η εκτέλεση:
  `cmd /c "python test_cases.py > run.log 2>&1"`. Μη διακόψεις την εκτέλεση πριν εμφανιστεί το SUMMARY.
