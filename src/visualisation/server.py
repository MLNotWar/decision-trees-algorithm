from flask import Flask, render_template, jsonify, abort


TEST_DATA = {
    "name": "root",
    "rule": "null",
    "children": [{
        "name": "5",
        "rule": "test"
    }, {
        "name": "2",
        "rule": "sunny",
        "children": [{
            "name": "no",
            "rule": "high",
        }, {
            "name": "yes",
            "rule": "normal",
        }]
    }, {
        "name": "yes",
        "rule": "overcast",
    }, {
        "name": "3",
        "rule": "rainy",
        "children": [{
            "name": "no",
            "rule": "TRUE",
        }, {
            "name": "yes",
            "rule": "FALSE",
        }]
    }]
}


def tag(data):
    def walk(_node):
        yield _node

        if "children" in _node:
            for _child in _node["children"]:
                yield from walk(_child)

    n = 0
    for node in walk(data):
        if "id" not in node:
            node["id"] = n
            n += 1

    return data


def main(data, **options):
    app = Flask("visualisation")
    data = {str(k): tag(v) for k, v in data.items()}

    @app.route("/show/<label>")
    def index(label):
        return render_template("app.html", label=label) if label in data else abort(404)

    @app.route("/data/<label>")
    def _(label):
        return jsonify(data[label]) if label in data else abort(404)

    app.run(**options)


if __name__ == "__main__":
    main({"test": TEST_DATA})
