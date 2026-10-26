# XAI Framework

Πλαίσιο εκτέλεσης μεθόδων επεξηγήσιμης τεχνητής νοημοσύνης (XAI) πάνω σε μοντέλα
sklearn και PyTorch. Υποστηρίζει 9 μεθόδους, για classification και regression,
με τοπικές (local) και καθολικές (global) επεξηγήσεις.

## Μέθοδοι

| Μέθοδος | Κλειδί στο `main.py` | Σημειώσεις |
|---|---|---|
| LIME | `lime` | Local explanations |
| TreeSHAP | `tree_shap` | Local και global explanations |
| KernelSHAP | `kernel_shap` | Local και global explanations· το global κομμάτι μπορεί να είναι αργό |
| Counterfactuals (DiCE) | `dice` | Local explanations |
| Anchors | `anchors` | Μόνο classification |
| ALE | `ale` | Global explanations |
| Permutation Feature Importance | `pfi` | Global explanations |
| PDP / ICE | `pdp_ice` | PDP: global, ICE: local |
| Integrated Gradients | `integrated_gradients` | Local attributions και global feature importance· μπορεί να χρησιμοποιεί surrogate για μη διαφορίσιμα μοντέλα |

## Απαιτήσεις

- **Python 3.11.** Το project δοκιμάστηκε με αυτή την έκδοση. Με άλλες εκδόσεις
  δεν είναι εγγυημένο ότι θα εγκατασταθούν όλα τα πακέτα (ειδικά τα `alibi` και `dice_ml`).
- Σύνδεση στο internet για την εγκατάσταση των πακέτων.
- Περίπου 3 GB ελεύθερου χώρου (το `torch` και οι εξαρτήσεις του είναι μεγάλα).

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

Τρέχει και τις 9 μεθόδους σε όλα τα 8 cases και τυπώνει ένα **SUMMARY** στο τέλος.
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

### Τι σημαίνει το SUMMARY

| Ένδειξη | Σημασία |
|---|---|
| `OK` | Η μέθοδος ολοκληρώθηκε και παρήγαγε αρχεία. |
| `OK (no files - skipped?)` | Η μέθοδος παραλείφθηκε σκόπιμα. Συμβαίνει στα **Anchors** σε regression, γιατί υποστηρίζονται μόνο για classification. Δεν είναι σφάλμα. |
| `N/A` | Δεν εφαρμόζεται. Στα cases με PyTorch NN τρέχει μόνο το **Integrated Gradients**, γιατί ένα απλό `nn.Module` δεν έχει `predict()`. Δεν είναι σφάλμα. |
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

Η εκτέλεση γίνεται μέσω του `main.py`, το οποίο υποστηρίζει ορίσματα γραμμής εντολών για την επιλογή dataset, μοντέλου, target, task, scope και XAI μεθόδων.

### Βασική εκτέλεση

```powershell
python main.py
```

Χρησιμοποιεί τις προεπιλεγμένες ρυθμίσεις του `config.py`.

### Εκτέλεση με custom dataset και μοντέλο

```powershell
python main.py --data test_dataset.csv --model test_model.joblib --target price --task regression --scope both
```

Στο παράδειγμα χρησιμοποιούνται:
- `--data`: διαδρομή του dataset.
- `--model`: διαδρομή του αποθηκευμένου μοντέλου.
- `--target`: όνομα της μεταβλητής-στόχου.
- `--task`: τύπος προβλήματος, `classification` ή `regression`.
- `--scope`: τύπος εξήγησης, `local`, `global` ή `both`.

Όταν δοθεί κάποιο από τα ορίσματα `--data`, `--model`, `--target` ή `--task`, η εκτέλεση γίνεται σε custom mode. Τα ορίσματα που παραλείπονται χρησιμοποιούν τις αντίστοιχες προεπιλεγμένες ρυθμίσεις του `config.py`.

### Επιλογή συγκεκριμένων XAI μεθόδων

Το όρισμα `--methods` επιτρέπει την επιλογή μίας ή περισσότερων μεθόδων. Οι διαθέσιμες τιμές είναι:

- `lime`
- `tree_shap`
- `kernel_shap`
- `dice`
- `anchors`
- `ale`
- `pfi`
- `pdp_ice`
- `integrated_gradients`

Παράδειγμα εκτέλεσης μόνο των LIME και TreeSHAP:

```powershell
python main.py --data test_dataset.csv --model test_model.joblib --target price --task regression --scope both --methods lime tree_shap
```

Παράδειγμα εκτέλεσης μόνο της ALE:

```powershell
python main.py --data test_dataset.csv --model test_model.joblib --target price --task regression --scope global --methods ale
```

Αν παραλειφθεί το `--methods`, χρησιμοποιείται η λίστα `xai_methods` από το `config.py`.

### Αποτελέσματα

Τα αποτελέσματα αποθηκεύονται στον φάκελο `outputs/`, σύμφωνα με τις ρυθμίσεις του `config.py`. Σε κάθε εκτέλεση του `main.py`, ο υπάρχων φάκελος αποτελεσμάτων καθαρίζεται πριν ξεκινήσει η παραγωγή νέων αποτελεσμάτων.

Στο τέλος της εκτέλεσης εμφανίζονται:
- Το task και το scope.
- Οι επιλεγμένες XAI μέθοδοι.
- Η διαδρομή του φακέλου αποτελεσμάτων.
- Ο αριθμός των explanations που επέστρεψε κάθε μέθοδος.

**Σημείωση:** Δεν υποστηρίζουν όλες οι μέθοδοι κάθε συνδυασμό task και scope. Για παράδειγμα, τα Anchors υποστηρίζονται μόνο για classification, ενώ τα ALE και PFI παράγουν global εξηγήσεις. Οι περιορισμοί αυτοί μπορεί να οδηγήσουν σε παράλειψη συγκεκριμένης εξήγησης χωρίς να σημαίνει απαραίτητα ότι απέτυχε ολόκληρη η εκτέλεση.

## Αναπαραγωγιμότητα

- Το `random_state` ορίζεται κεντρικά και περνάει σε όλες τις μεθόδους.
- Τα **Anchors** χρησιμοποιούν το ίδιο seed, οπότε δύο εκτελέσεις δίνουν τον ίδιο κανόνα.

## Συχνά προβλήματα

- **Το PowerShell μπλοκάρει το `activate`** ("running scripts is disabled"):
  τρέξε `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` και ξαναδοκίμασε.
- **Η έξοδος γεμίζει προειδοποιήσεις του sklearn** (`delayed ... Parallel`) και η εκτέλεση
  είναι πολύ αργή: έλεγξε ότι το `scikit-learn` είναι 1.8.0 (`python -c "import sklearn; print(sklearn.__version__)"`).
- **Θέλεις log αρχείο:** χρησιμοποίησε το `cmd` για να μη χαθεί η έξοδος αν διακοπεί η εκτέλεση:
  `cmd /c "python test_cases.py > run.log 2>&1"`. Μη διακόψεις την εκτέλεση πριν εμφανιστεί το SUMMARY.
