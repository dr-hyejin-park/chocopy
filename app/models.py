from app import db

class Getrecommender(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    platform = db.Column(db.String(200), nullable=False)
    modelType = db.Column(db.Text(), nullable=False)
    item = db.Column(db.String(200), nullable=False)
    eventTime = db.Column(db.DateTime(), nullable=False)



class Putrecommend(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recommend_id = db.Column(db.Integer, db.ForeignKey('getrecommender.id', ondelete='CASCADE'))
    recommend = db.relationship('Getrecommender', backref=db.backref('item_set'))
    item = db.Column(db.String(200), nullable=False)
    eventTime = db.Column(db.DateTime(), nullable=False)


class Useritemmodel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(150), unique=True, nullable=False)
    item_id = db.Column(db.String(200), nullable=False)
    model_a = db.Column(db.String(120), nullable=False)
    model_b = db.Column(db.String(120), nullable=False)
    model_c = db.Column(db.String(120), nullable=False)


class sims_reco(db.Model):
    index = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, unique=True, nullable=False)
    # item_id = db.Column(db.Integer, nullable=False)
    platform = db.Column(db.String(200), nullable=False)
    modelType = db.Column(db.Text(), nullable=False)
    recommender_id = db.Column(db.Integer, db.ForeignKey('getrecommender.id', ondelete='CASCADE'))
    top1=db.Column(db.String(120), nullable=False)
    top2=db.Column(db.String(120), nullable=False)
    top3=db.Column(db.String(120), nullable=False)
    top4=db.Column(db.String(120), nullable=False)
    top5=db.Column(db.String(120), nullable=False)
    top6=db.Column(db.String(120), nullable=False)
    top7=db.Column(db.String(120), nullable=False)
    top8=db.Column(db.String(120), nullable=False)
    top9=db.Column(db.String(120), nullable=False)
    top10=db.Column(db.String(120), nullable=False)



class user_history(db.Model):
    index = db.Column(db.Integer, primary_key=True)
    user_name=db.Column(db.String(120), nullable=False)
    user_id=db.Column(db.Integer, nullable=False)
    item_name=db.Column(db.String(120), nullable=False)
    item_id=db.Column(db.Integer, nullable=False)
    timestamp=db.Column(db.DateTime(), nullable=False)
    