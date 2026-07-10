from flask import Flask, request, jsonify, redirect,render_template
from database import initialize_database, get_connection
from utils import generate_short_code
import validators

app = Flask(__name__)

initialize_database()

@app.route("/app")
def frontend():
    return render_template("index.html")

@app.route("/")
def home():
    return jsonify({
        "success": True,
        "message": "Welcome to URL Shortener API"
    })


@app.route("/shorten", methods=["POST"])
def shorten_url():

    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({
            "success": False,
            "message": "URL is required"
        }), 400

    original_url = data["url"]

    if not validators.url(original_url):
        return jsonify({
            "success": False,
            "message": "Invalid URL"
        }), 400

    conn = get_connection()
    cursor = conn.cursor()

    # Check if URL already exists
    cursor.execute(
        "SELECT short_code FROM urls WHERE original_url=?",
        (original_url,)
    )

    existing = cursor.fetchone()

    if existing:
        conn.close()

        return jsonify({
            "success": True,
            "message": "URL already shortened",
            "short_code": existing["short_code"],
            "short_url": f"http://127.0.0.1:5000/{existing['short_code']}"
        }), 200

    # Generate unique short code
    while True:

        short_code = generate_short_code()

        cursor.execute(
            "SELECT id FROM urls WHERE short_code=?",
            (short_code,)
        )

        if cursor.fetchone() is None:
            break

    cursor.execute(
        "INSERT INTO urls(short_code, original_url) VALUES (?, ?)",
        (short_code, original_url)
    )

    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": "Short URL created successfully",
        "original_url": original_url,
        "short_code": short_code,
        "short_url": f"http://127.0.0.1:5000/{short_code}"
    }), 201


@app.route("/<short_code>")
def redirect_url(short_code):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM urls WHERE short_code=?",
        (short_code,)
    )

    url = cursor.fetchone()

    if not url:
        conn.close()
        return jsonify({
            "success": False,
            "message": "Short URL not found"
        }), 404

    cursor.execute(
        "UPDATE urls SET clicks = clicks + 1 WHERE short_code=?",
        (short_code,)
    )

    conn.commit()
    conn.close()

    return redirect(url["original_url"])


@app.route("/analytics/<short_code>")
def analytics(short_code):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM urls WHERE short_code=?",
        (short_code,)
    )

    url = cursor.fetchone()

    conn.close()

    if not url:
        return jsonify({
            "success": False,
            "message": "Short URL not found"
        }), 404

    return jsonify({
        "success": True,
        "original_url": url["original_url"],
        "short_code": url["short_code"],
        "clicks": url["clicks"],
        "created_at": url["created_at"]
    })


@app.route("/all")
def get_all_urls():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM urls")

    rows = cursor.fetchall()

    conn.close()

    urls = []

    for row in rows:
        urls.append({
            "short_code": row["short_code"],
            "original_url": row["original_url"],
            "clicks": row["clicks"],
            "created_at": row["created_at"]
        })

    return jsonify(urls)


@app.route("/stats")
def stats():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM urls")
    total_urls = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(clicks) FROM urls")
    total_clicks = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        "success": True,
        "total_urls": total_urls,
        "total_clicks": total_clicks or 0
    })


@app.route("/delete/<short_code>", methods=["DELETE"])
def delete_url(short_code):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM urls WHERE short_code=?",
        (short_code,)
    )

    conn.commit()

    if cursor.rowcount == 0:
        conn.close()

        return jsonify({
            "success": False,
            "message": "Short URL not found"
        }), 404

    conn.close()

    return jsonify({
        "success": True,
        "message": "Short URL deleted successfully"
    })


if __name__ == "__main__":
    app.run(debug=True)