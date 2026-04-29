from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json, os, math

# ══════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════
app = Flask(__name__)
CORS(app)

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'campus.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ══════════════════════════════════════
#  MODÈLES
# ══════════════════════════════════════
MODULES_VALIDES = ['etudes', 'budget', 'alim', 'sante', 'mobilite', 'numerique']

class Reponse(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    session   = db.Column(db.String(64), nullable=False, index=True)
    module    = db.Column(db.String(32), nullable=False)
    donnees   = db.Column(db.Text, nullable=False)   # JSON
    cree_le   = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id'      : self.id,
            'session' : self.session,
            'module'  : self.module,
            'donnees' : json.loads(self.donnees),
            'cree_le' : self.cree_le.strftime('%d/%m/%Y %H:%M')
        }

# ══════════════════════════════════════
#  UTILITAIRES STATISTIQUES (Python pur)
# ══════════════════════════════════════
def moyenne(valeurs):
    v = [float(x) for x in valeurs if x is not None and str(x).strip() != '']
    return round(sum(v) / len(v), 2) if v else 0

def mediane(valeurs):
    v = sorted([float(x) for x in valeurs if x is not None and str(x).strip() != ''])
    if not v: return 0
    n = len(v)
    mid = n // 2
    return round((v[mid-1] + v[mid]) / 2, 2) if n % 2 == 0 else round(v[mid], 2)

def ecart_type(valeurs):
    v = [float(x) for x in valeurs if x is not None and str(x).strip() != '']
    if len(v) < 2: return 0
    m = sum(v) / len(v)
    variance = sum((x - m) ** 2 for x in v) / len(v)
    return round(math.sqrt(variance), 2)

def variance(valeurs):
    v = [float(x) for x in valeurs if x is not None and str(x).strip() != '']
    if len(v) < 2: return 0
    m = sum(v) / len(v)
    return round(sum((x - m) ** 2 for x in v) / len(v), 2)

def q1_q3(valeurs):
    v = sorted([float(x) for x in valeurs if x is not None and str(x).strip() != ''])
    if not v: return 0, 0
    return round(v[len(v)//4], 2), round(v[len(v)*3//4], 2)

def stats_variable(valeurs):
    v = [float(x) for x in valeurs if x is not None and str(x).strip() != '']
    if not v: return None
    s = sorted(v)
    q1, q3 = q1_q3(v)
    return {
        'n'        : len(v),
        'min'      : round(s[0], 2),
        'max'      : round(s[-1], 2),
        'moyenne'  : moyenne(v),
        'mediane'  : mediane(v),
        'ecart_type': ecart_type(v),
        'variance' : variance(v),
        'q1'       : q1,
        'q3'       : q3,
    }

# ── Régression linéaire simple (OLS) ──
def regression_simple(x_vals, y_vals):
    n = len(x_vals)
    if n < 3:
        return None
    mx = sum(x_vals) / n
    my = sum(y_vals) / n
    sxy = sum((x_vals[i]-mx)*(y_vals[i]-my) for i in range(n))
    sxx = sum((x_vals[i]-mx)**2 for i in range(n))
    if sxx == 0:
        return None
    b1 = sxy / sxx
    b0 = my - b1 * mx
    y_pred = [b0 + b1*x for x in x_vals]
    sse = sum((y_vals[i]-y_pred[i])**2 for i in range(n))
    sst = sum((y_vals[i]-my)**2 for i in range(n))
    r2 = 1 - sse/sst if sst else 0
    rmse = math.sqrt(sse/n)
    r = math.sqrt(abs(r2)) * (1 if b1 >= 0 else -1)
    return {
        'b0'    : round(b0, 4),
        'b1'    : round(b1, 4),
        'r2'    : round(r2, 4),
        'r'     : round(r, 4),
        'rmse'  : round(rmse, 4),
        'sse'   : round(sse, 4),
        'sst'   : round(sst, 4),
        'n'     : n,
        'points': [{'x': round(x,2), 'y': round(y,2)} for x,y in zip(x_vals, y_vals)],
        'droite': [
            {'x': round(min(x_vals),2), 'y': round(b0+b1*min(x_vals),2)},
            {'x': round(max(x_vals),2), 'y': round(b0+b1*max(x_vals),2)},
        ]
    }

# ── Régression multiple (OLS matricielle) ──
def mat_mul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def mat_inv(M):
    n = len(M)
    A = [row[:] for row in M]
    I = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        mx_val, mx_i = abs(A[c][c]), c
        for r in range(c+1, n):
            if abs(A[r][c]) > mx_val:
                mx_val, mx_i = abs(A[r][c]), r
        A[c], A[mx_i] = A[mx_i], A[c]
        I[c], I[mx_i] = I[mx_i], I[c]
        piv = A[c][c]
        if abs(piv) < 1e-12: return None
        for j in range(n):
            A[c][j] /= piv; I[c][j] /= piv
        for r in range(n):
            if r == c: continue
            f = A[r][c]
            for j in range(n):
                A[r][j] -= f*A[c][j]; I[r][j] -= f*I[c][j]
    return I

def regression_multiple(X_data, y_data, x_noms):
    n = len(X_data)
    p = len(x_noms)
    if n < p + 2: return None
    X = [[1.0] + list(row) for row in X_data]
    Xt = [[X[r][c] for r in range(n)] for c in range(p+1)]
    XtX = mat_mul(Xt, X)
    XtXi = mat_inv(XtX)
    if not XtXi: return None
    Xty = [sum(Xt[i][j]*y_data[j] for j in range(n)) for i in range(p+1)]
    B = [sum(XtXi[i][j]*Xty[j] for j in range(p+1)) for i in range(p+1)]
    y_pred = [sum(B[i]*X[r][i] for i in range(p+1)) for r in range(n)]
    my = sum(y_data)/n
    sse = sum((y_data[i]-y_pred[i])**2 for i in range(n))
    sst = sum((y_data[i]-my)**2 for i in range(n))
    r2 = 1 - sse/sst if sst else 0
    r2a = 1 - (1-r2)*(n-1)/(n-p-1)
    rmse = math.sqrt(sse/n)
    return {
        'b0'      : round(B[0], 4),
        'betas'   : [{'nom': x_noms[i], 'val': round(B[i+1], 4)} for i in range(p)],
        'r2'      : round(r2, 4),
        'r2_adj'  : round(r2a, 4),
        'rmse'    : round(rmse, 4),
        'n'       : n,
        'reel_vs_pred': [{'reel': round(y_data[i],2), 'pred': round(y_pred[i],2)} for i in range(n)]
    }

# ── K-Means ──
def eucl(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))

def kmeans(data, k=3, max_iter=100):
    import random
    if len(data) < k: return None
    # Normalisation min-max
    mins = [min(r[j] for r in data) for j in range(len(data[0]))]
    maxs = [max(r[j] for r in data) for j in range(len(data[0]))]
    Z = [[(r[j]-mins[j])/(maxs[j]-mins[j]) if maxs[j]!=mins[j] else 0 for j in range(len(r))] for r in data]
    # K-Means++ init
    cents = [Z[random.randint(0, len(Z)-1)]]
    while len(cents) < k:
        dists = [min(eucl(p, c) for c in cents) for p in Z]
        s = sum(dists)
        r = random.random() * s
        for i, d in enumerate(dists):
            r -= d
            if r <= 0:
                cents.append(Z[i][:])
                break
    labels = [0] * len(Z)
    for _ in range(max_iter):
        new_labels = [min(range(k), key=lambda ki: eucl(p, cents[ki])) for p in Z]
        if new_labels == labels: break
        labels = new_labels
        for ki in range(k):
            pts = [Z[i] for i in range(len(Z)) if labels[i]==ki]
            if pts:
                cents[ki] = [sum(p[j] for p in pts)/len(pts) for j in range(len(Z[0]))]
    wcss = sum(eucl(Z[i], cents[labels[i]])**2 for i in range(len(Z)))
    sizes = [labels.count(ki) for ki in range(k)]
    # Centroïdes en échelle originale
    cents_orig = [[round(c[j]*(maxs[j]-mins[j])+mins[j], 2) for j in range(len(c))] for c in cents]
    return {
        'labels'  : labels,
        'wcss'    : round(wcss, 4),
        'sizes'   : sizes,
        'centroides': cents_orig,
        'projection': [{'x': round(Z[i][0],3), 'y': round(Z[i][1] if len(Z[i])>1 else Z[i][0],3), 'cluster': labels[i]} for i in range(len(Z))]
    }

# ══════════════════════════════════════
#  ROUTES — PAGES
# ══════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

# ══════════════════════════════════════
#  ROUTES — API
# ══════════════════════════════════════

# ── Santé ──
@app.route('/api/sante')
def api_sante():
    return jsonify({'status': 'ok', 'message': 'Campus Insight API - Python/Flask'})

# ── Enregistrer une réponse ──
@app.route('/api/reponses', methods=['POST'])
def ajouter_reponse():
    data = request.get_json()
    session = data.get('session')
    module  = data.get('module')
    donnees = data.get('donnees')

    if not session or not module or not donnees:
        return jsonify({'erreur': 'session, module et donnees sont requis'}), 400
    if module not in MODULES_VALIDES:
        return jsonify({'erreur': f'Module invalide. Valeurs acceptées : {MODULES_VALIDES}'}), 400

    rep = Reponse(session=session, module=module, donnees=json.dumps(donnees))
    db.session.add(rep)
    db.session.commit()
    return jsonify({'succes': True, 'id': rep.id}), 201

# ── Récupérer les réponses d'une session ──
@app.route('/api/reponses/<session_id>')
def get_reponses(session_id):
    reps = Reponse.query.filter_by(session=session_id).all()
    groupees = {m: [] for m in MODULES_VALIDES}
    for r in reps:
        d = json.loads(r.donnees)
        d['_date'] = r.cree_le.strftime('%d/%m/%Y %H:%M')
        groupees[r.module].append(d)
    return jsonify({'succes': True, 'donnees': groupees, 'total': len(reps)})

# ── Supprimer un module d'une session ──
@app.route('/api/reponses/<session_id>/<module>', methods=['DELETE'])
def supprimer_module(session_id, module):
    nb = Reponse.query.filter_by(session=session_id, module=module).delete()
    db.session.commit()
    return jsonify({'succes': True, 'supprimees': nb})

# ── Supprimer toutes les réponses d'une session ──
@app.route('/api/reponses/<session_id>', methods=['DELETE'])
def supprimer_session(session_id):
    nb = Reponse.query.filter_by(session=session_id).delete()
    db.session.commit()
    return jsonify({'succes': True, 'supprimees': nb})

# ── Statistiques globales ──
@app.route('/api/stats/global')
def stats_global():
    stats = {}
    for mod in MODULES_VALIDES:
        reps = Reponse.query.filter_by(module=mod).all()
        if not reps:
            stats[mod] = None
            continue
        all_data = [json.loads(r.donnees) for r in reps]
        cles_num = [k for k, v in all_data[0].items() if isinstance(v, (int, float))]
        stats[mod] = {}
        for k in cles_num:
            vals = [d.get(k) for d in all_data if isinstance(d.get(k), (int, float))]
            stats[mod][k] = stats_variable(vals)
    return jsonify({'succes': True, 'stats': stats})

# ── Compteurs ──
@app.route('/api/stats/compteurs')
def compteurs():
    sessions = db.session.query(Reponse.session).distinct().count()
    total    = Reponse.query.count()
    return jsonify({'succes': True, 'sessions': sessions, 'total': total})

# ══════════════════════════════════════
#  ROUTES — ANALYSE (Python côté serveur)
# ══════════════════════════════════════

# ── Statistiques descriptives d'une session ──
@app.route('/api/analyse/<session_id>')
def analyse_session(session_id):
    reps = Reponse.query.filter_by(session=session_id).all()
    if not reps:
        return jsonify({'erreur': 'Aucune donnée pour cette session'}), 404

    groupees = {m: [] for m in MODULES_VALIDES}
    for r in reps:
        groupees[r.module].append(json.loads(r.donnees))

    resultats = {}
    for mod, donnees in groupees.items():
        if not donnees: continue
        cles_num = [k for k, v in donnees[0].items() if isinstance(v, (int, float))]
        resultats[mod] = {}
        for k in cles_num:
            vals = [d.get(k) for d in donnees if isinstance(d.get(k), (int, float))]
            st = stats_variable(vals)
            if st: resultats[mod][k] = st

    return jsonify({'succes': True, 'stats': resultats})

# ── Régression linéaire simple ──
@app.route('/api/ml/regression-simple', methods=['POST'])
def ml_regression_simple():
    data = request.get_json()
    session_id = data.get('session')
    x_nom = data.get('x')
    y_nom = data.get('y')

    if not session_id or not x_nom or not y_nom:
        return jsonify({'erreur': 'session, x et y requis'}), 400
    if x_nom == y_nom:
        return jsonify({'erreur': 'X et Y doivent être différents'}), 400

    reps = Reponse.query.filter_by(session=session_id).all()
    all_data = [json.loads(r.donnees) for r in reps]
    pts = [(d.get(x_nom), d.get(y_nom)) for d in all_data
           if isinstance(d.get(x_nom), (int,float)) and isinstance(d.get(y_nom), (int,float))]
    if len(pts) < 3:
        return jsonify({'erreur': 'Minimum 3 observations requises'}), 400

    x_vals = [p[0] for p in pts]
    y_vals = [p[1] for p in pts]
    resultat = regression_simple(x_vals, y_vals)
    if not resultat:
        return jsonify({'erreur': 'Impossible de calculer la régression'}), 400

    resultat.update({'x_nom': x_nom, 'y_nom': y_nom})
    return jsonify({'succes': True, 'resultat': resultat})

# ── Régression multiple ──
@app.route('/api/ml/regression-multiple', methods=['POST'])
def ml_regression_multiple():
    data = request.get_json()
    session_id = data.get('session')
    y_nom  = data.get('y')
    x_noms = data.get('xs', [])

    if not session_id or not y_nom or not x_noms:
        return jsonify({'erreur': 'session, y et xs requis'}), 400
    x_noms = [x for x in x_noms if x != y_nom]
    if not x_noms:
        return jsonify({'erreur': 'Sélectionnez au moins une variable X différente de Y'}), 400

    reps = Reponse.query.filter_by(session=session_id).all()
    all_data = [json.loads(r.donnees) for r in reps]

    X_data, y_data = [], []
    for d in all_data:
        row_x = [d.get(x) for x in x_noms]
        row_y = d.get(y_nom)
        if all(isinstance(v, (int,float)) for v in row_x) and isinstance(row_y, (int,float)):
            X_data.append([float(v) for v in row_x])
            y_data.append(float(row_y))

    if len(X_data) < len(x_noms) + 2:
        return jsonify({'erreur': 'Pas assez de données'}), 400

    resultat = regression_multiple(X_data, y_data, x_noms)
    if not resultat:
        return jsonify({'erreur': 'Matrice singulière — vérifiez les variables'}), 400

    resultat.update({'y_nom': y_nom, 'x_noms': x_noms})
    return jsonify({'succes': True, 'resultat': resultat})

# ── K-Means ──
@app.route('/api/ml/kmeans', methods=['POST'])
def ml_kmeans():
    data = request.get_json()
    session_id = data.get('session')
    variables  = data.get('variables', [])
    k          = int(data.get('k', 3))
    max_iter   = int(data.get('max_iter', 100))

    if not session_id or not variables:
        return jsonify({'erreur': 'session et variables requis'}), 400

    reps = Reponse.query.filter_by(session=session_id).all()
    all_data = [json.loads(r.donnees) for r in reps]
    mat = []
    for d in all_data:
        row = [d.get(v) for v in variables]
        if all(isinstance(v, (int,float)) for v in row):
            mat.append([float(v) for v in row])

    if len(mat) < k:
        return jsonify({'erreur': f'Pas assez de données ({len(mat)} obs pour {k} clusters)'}), 400

    resultat = kmeans(mat, k=k, max_iter=max_iter)
    if not resultat:
        return jsonify({'erreur': 'Impossible de calculer K-Means'}), 400

    resultat.update({'variables': variables, 'k': k})
    return jsonify({'succes': True, 'resultat': resultat})

# ══════════════════════════════════════
#  LANCEMENT
# ══════════════════════════════════════
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print(f"🚀 Campus Insight — Python/Flask démarré sur http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
