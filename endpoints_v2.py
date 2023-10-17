import os
import uuid

from botocore.exceptions import ClientError
from flask import g, request, send_file
from flask_restx import Namespace, Resource, fields, inputs, marshal
from oracle_db import Session
from oracle_db.schema.Artifact import (
    Artifact,
    ArtifactAnswer,
    ArtifactOrder,
    ArtifactProgrammingResource,
    ArtifactTestCase,
    Metadata,
    UserArtifact,
    UserHistoryV2,
)
from oracle_db.schema.Topic import Topic
from sqlalchemy import Date, cast, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload, subqueryload
from werkzeug.datastructures import FileStorage

from oracle.api import set_user_history_id
from oracle.api.ml_endpoints import output_coding_feedback_response
from oracle.api.util import (
    clear_cache_and_update_artifact,
    clear_cache_for_deprecated,
    do_paging,
    filter_based_on_search_params,
    filter_cached_based_on_search_params,
    filter_cursor,
    is_valid_uuid,
    json_to_csv_file,
    populate_artifacts,
    populate_user_artifacts,
)
from oracle.config import config
from oracle.core import ArtifactType, db
from oracle.core.cache import RedisClusterJson
from oracle.core.coding_feedback import CodingFeedbackManager
from oracle.core.db import get_artifact_by_guid, get_artifacts
from oracle.core.exceptions import NotAResource, NotFound
from oracle.core.programming_exercises import (
    update_programming_exercise_test_case,
    update_programming_resource,
)
from oracle.core.request_parsers import (
    add_artifact_search_params,
    add_pagination_params,
)
from oracle.core.s3 import retrieve_programming_exercise_s3_content
from oracle.core.test_cases import (
    add_record_property_to_code,
    replace_all_feedback_strings_with_guids,
)
from oracle.core.util import (
    delete_artifact_reference_fields,
    join_s3_path,
    read_s3_file,
    set_artifact_reference_depths,
    set_artifact_reference_fields,
    split_s3_path,
    split_s3_prefix_path,
    upload_feedback_mappings_from_test_file_to_db,
    upload_file_to_aws,
)


class NestedArtifactField(fields.Raw):
    """
    A class for fields that are a guid but may optionally be artifact objects if the client requests a specific depth
    """

    field_name = ""
    join_table_class = None

    def format(self, value):
        if value and hasattr(value[0], "depth") and value[0].depth > 0:
            formatted = []
            for v in value:
                artifact = Session.object_session(v).get(Artifact, getattr(v, self.field_name))
                [artifact] = set_artifact_reference_depths([artifact], v.depth - 1)
                formatted.append(marshal(artifact, ns.models["artifact"]))
            return formatted
        else:
            return [Session.object_session(v).get(Artifact, getattr(v, self.field_name)).guid for v in value]

    def schema(self):
        schema = super(NestedArtifactField, self).schema()
        schema["type"] = "array"
        schema["items"] = fields.String().schema()
        return schema


class SortedNestedArtifactField(NestedArtifactField):
    """
    A class for fields that are a guid but may optionally be artifact objects if the client requests a specific depth
    """

    field_name = ""
    join_table_class = None

    def format(self, value):
        if value and hasattr(value[0], "depth") and value[0].depth > 0:
            formatted = []
            for v in sorted(value, key=lambda x: x.order):
                artifact = Session.object_session(v).get(Artifact, getattr(v, self.field_name))
                [artifact] = set_artifact_reference_depths([artifact], v.depth - 1)
                formatted.append(marshal(artifact, ns.models["artifact"]))
            return formatted
        else:
            return [
                Session.object_session(v).get(Artifact, getattr(v, self.field_name)).guid
                for v in sorted(value, key=lambda x: x.order)
            ]


class ArtifactOrderField(SortedNestedArtifactField):
    join_table_class = ArtifactOrder
    field_name = "child_id"


class ArtifactGuid(fields.Raw):
    __schema_type__ = "string"

    def format(self, value):
        if value:
            return value.guid
        return None


class ArtifactAnswerField(NestedArtifactField):
    field_name = "answer_id"
    join_table_class = ArtifactAnswer


class ArtifactTestCaseField(NestedArtifactField):
    field_name = "test_case_id"
    join_table_class = ArtifactTestCase


class ArtifactNavigational(fields.Raw):
    __schema_type__ = "object"

    def format(self, value):
        choices = [
            {
                "title": v.choice,
                "order": v.order,
                "artifact_id": v.navigation.guid if v.navigation_id else None,
            }
            for v in value
        ]

        return choices


class ArtifactProgrammingResourceField(NestedArtifactField):
    field_name = "resource_id"
    join_table_class = ArtifactProgrammingResource


ns = Namespace(
    "endpoints",
    "APIs for retrieving oracle data.",
)


artifact_model = ns.model(
    "artifact",
    {
        "id": fields.String(readonly=True, attribute="guid"),
        "type": fields.String,
        "artifacts": ArtifactOrderField(required=False, default=[]),
        "answers": ArtifactAnswerField(required=False, default=[]),
        "content": fields.String,
        "test_cases": ArtifactTestCaseField(required=False, default=[]),
        "choices": ArtifactNavigational(required=False, default=[]),
        "in_line": fields.Boolean(required=False, default=False),
        "boilerplate_code": ArtifactGuid(required=False, attribute="boilerplate_code_artifact", default=None),
        "programming_resources": ArtifactProgrammingResourceField(required=False, default=[]),
        "topics": fields.List(fields.String(attribute="topic"), required=False, default=[]),
        "creation_time": fields.DateTime(readonly=True),
        "creator": fields.String,
        "title": fields.String,
        "deprecated": fields.Boolean(readonly=True),
    },
)
redis_cluster = RedisClusterJson()

user_model = ns.model(
    "user",
    {"id": fields.String(readonly=True, attribute="guid"), "name": fields.String},
)
ml_data_model = ns.model(
    "ml_data",
    {
        "version": fields.String,
        "name": fields.String,
        "input": fields.Raw,
        "output": fields.Raw,
    },
)

user_artifact_model = ns.model(
    "user_artifact",
    {
        "id": fields.String(readonly=True, attribute="user_artifact_guid"),
        "content": fields.String,
        "response_to": fields.String(attribute="artifact_guid"),
    },
)

user_history_user_artifact_model = ns.model(
    "user_artifact",
    {
        "guid": fields.String(readonly=True),
        "content": fields.String,
        "response_to": fields.String,
    },
)

user_history_model = ns.model(
    "user_history",
    {
        "id": fields.Integer(readonly=True),
        "user": fields.String,
        "ml_data": fields.Nested(ml_data_model),
        "coding_feedback_response_id": fields.String,
        "coding_feedback_response": fields.Nested(output_coding_feedback_response),
        "timestamp": fields.DateTime(readonly=True),
        "artifact_id": fields.String,
        "user_artifact_id": fields.String,
        "session": fields.String,
        "sequence": fields.Integer(readonly=True),
    },
)

input_user_history_model = ns.model(
    "user_history",
    {
        "id": fields.Integer(readonly=True),
        "user": fields.String,
        "ml_data": fields.Nested(ml_data_model),
        "coding_feedback_response_id": fields.String,
        "timestamp": fields.DateTime(readonly=True),
        "artifact_id": fields.String,
        "user_artifact_id": fields.String,
        "session": fields.String,
        "sequence": fields.Integer(readonly=True),
    },
)

topic_model = ns.model("topic", {"name": fields.String(readonly=True)})

artifact_parser = ns.parser()
artifact_parser.add_argument(
    "type",
    type=str,
    choices=[artifact_type.value for artifact_type in ArtifactType],
    required=True,
)
artifact_parser.add_argument("artifacts", type=list, default=None, location="json")
artifact_parser.add_argument("answers", type=list, default=None, location="json")
artifact_parser.add_argument("content", type=str, required=True)
artifact_parser.add_argument("in_line", type=bool, required=False, default=False)
artifact_parser.add_argument("test_cases", type=list, default=None, location="json")
artifact_parser.add_argument("choices", type=list, default=None, location="json")
artifact_parser.add_argument("boilerplate_code", type=str, location="json")
artifact_parser.add_argument("programming_resources", type=list, default=None, location="json")
artifact_parser.add_argument("topics", type=list, default=None, location="json")
artifact_parser.add_argument("creator", type=str, default=None, location="json")
artifact_parser.add_argument("title", type=str, default=None, location="json")

put_artifact_parser = artifact_parser.copy()
put_artifact_parser.replace_argument(
    "type",
    type=str,
    choices=[artifact_type.value for artifact_type in ArtifactType],
    location="json",
)
put_artifact_parser.replace_argument("content", type=str, location="json")
put_artifact_parser.add_argument("clear_cache", type=inputs.boolean, location="values", default=True)

artifact_doc_params = {
    "type": f"One of {[_type.value for _type in ArtifactType]}",
    "content": "The actual text, etc",
    "artifacts": {
        "description": "A list of artifact ids that will be shown as part of this artifact",
        "type": "array",
    },
    "answers": {
        "description": "A list of artifact ids that will be used on correct answers",
        "type": "array",
    },
    "in_line": {
        "description": "A flag that will combine a student's submission with the exercise's boilerplate code",
        "type": "bool",
    },
    "test_cases": {
        "description": "The test cases to run (list of artifact ids), if this artifact is a programming exercise",
        "type": "array",
    },
    "choices": {
        "description": 'The choices will be shown to the user, if this artifact is a navigational. Should be an array of dictionary: [{"title": button, "order": number, "artifact_id": guid}]',
        "type": "array",
    },
    "programming_resources": {
        "description": "Additional resource artifacts needed for executing code.",
        "type": "array",
    },
}


get_artifact_parser = ns.parser()
get_artifact_parser.add_argument(
    "depth",
    type=int,
    default=0,
    help="How many child artifacts deep to go.",
    location="values",
)
get_artifact_parser.add_argument(
    "show_deprecated",
    type=inputs.boolean,
    default=False,
    help="Whether to show deprecated artifacts or not.",
    location="values",
)
get_artifact_parser = add_artifact_search_params(get_artifact_parser)
get_artifact_parser = add_pagination_params(get_artifact_parser)

get_multiple_artifact_parser_1 = ns.parser()
get_multiple_artifact_parser.add_argument(
    "depth",
    type=int,
    default=1,
    help="How many child artifacts deep to go.",
    location="values",
)
get_multiple_artifact_parser.add_argument(
    "artifact_id",
    type=str,
    default="",
    help="Artifact id",
    location="values",
)


@ns.route("/artifacts")
class Artifacts(Resource):
    @ns.expect(get_artifact_parser)
    def get(self):
        args = get_artifact_parser.parse_args()
        depth = args.pop("depth")
        show_deprecated = args.pop("show_deprecated")
        limit, offset = args.pop("limit"), args.pop("offset")
        if offset < 0:
            offset = 0
        artifacts = redis_cluster.get_all_artifacts(depth)
        if artifacts:
            artifacts = filter_cached_based_on_search_params(artifacts, args)
            if not show_deprecated:
                artifacts = [artifact for artifact in artifacts if not artifact["deprecated"]]
            if limit:
                artifacts, has_more = do_paging(artifacts, offset, limit)
            return artifacts, 200
        with Session(expire_on_commit=False) as session:
            all_artifacts = get_artifacts(session)
            artifacts = filter_cursor(all_artifacts, args)
            total_filtered = artifacts.count()
            if limit:
                artifacts = artifacts.offset(offset * limit).limit(limit)
            artifacts = artifacts.all()
            all_artifacts = all_artifacts.all()

            # need to ensure a strong reference exists to the cached artifacts otherwise we will send out unnecessary
            # SELECT statements
            g.all_artifacts = {a.id: a for a in all_artifacts}
            if not show_deprecated:
                artifacts = [a for a in artifacts if a.deprecated == False]
            artifacts = set_artifact_reference_depths(artifacts, depth)
            artifacts = marshal(artifacts, artifact_model)

            all_artifacts = set_artifact_reference_depths(all_artifacts, depth)
            redis_cluster.set_all_artifacts(depth, marshal(all_artifacts, artifact_model))

        return artifacts, 200

    @ns.expect(artifact_model)
    @ns.doc(params=artifact_doc_params)
    def post(self):
        data = artifact_parser.parse_args()
        artifact = Artifact(
            content=data["content"],
            type=data["type"],
            title=data["title"],
            creator=data["creator"],
            in_line=data["in_line"],
        )
        with Session(expire_on_commit=False) as session:
            session.add(artifact)
            try:
                set_artifact_reference_fields(session, data, artifact)
            except KeyError as e:
                session.rollback()
                session.close()
                return {"message": f"Artifact with guid {e} not found, no artifact with that guid exists."}, 400
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                session.close()
                return {
                    "message": "Unable to create the artifact, there is a foreign key specified that doesn't exist."
                }, 400
            marshalled = marshal(artifact, artifact_model)
            redis_cluster.clear_cache(redis_cluster.ALL_ARTIFACTS_PREFIX)

        return artifact.guid, 200


@ns.route("/artifacts/exercise/<content>")
class ArtifactByContent(Resource):
    @ns.doc(params={"content": {"description": "The content for which you want the artifact id"}})
    def get(self, content):
        artifact = None
        with Session() as session:
            artifact = (
                session.query(Artifact)
                .filter(
                    Artifact.content == content,
                    Artifact.type == "exercise",
                    Artifact.deprecated == False,
                )
                .first()
            )
            if artifact:
                artifact = marshal(artifact, artifact_model)
        if artifact:
            return artifact, 200
        return {"message": f"Unable to retrieve artifact: Exercise with content tite: {content} does not exist."}, 400


@ns.route("/artifact_list/")
class MultipleArtifact(Resource):
    @ns.expect(get_multiple_artifact_parser)
    def get(self):
        depth = get_multiple_artifact_parser.parse_args()["depth"]
        artifact_list = request.args.getlist("artifact_id")
        output_artifacts = []

        for artifact_id in artifact_list:
            artifact_id = artifact_id.strip()
            if not is_valid_uuid(artifact_id):
                return {"message": f"Invalid artifact id format: Artifact with id {artifact_id} is not valid."}, 400
            artifact = redis_cluster.get_artifact(artifact_id, depth)
            if artifact:
                output_artifacts.append(artifact)
            else:
                with Session() as session:
                    artifact = (
                        session.query(Artifact)
                        .options(joinedload(Artifact.boilerplate_code_artifact))
                        .filter(Artifact.guid == artifact_id)
                        .first()
                    )
                    if artifact:
                        [artifact] = set_artifact_reference_depths([artifact], depth=depth)
                        artifact = marshal(artifact, artifact_model)
                        redis_cluster.set_artifact(artifact_id, depth, artifact)
                if artifact:
                    output_artifacts.append(artifact)
                else:
                    return {"message": f"Invalid artifact id provided: No artifact with id {artifact_id} exists."}, 400

        return {"artifacts": output_artifacts}, 200


@ns.route("/artifacts/<artifact_id>")
class SingleArtifact(Resource):
    @ns.expect(get_artifact_parser)
    def get(self, artifact_id):
        depth = get_artifact_parser.parse_args()["depth"]
        artifact = redis_cluster.get_artifact(artifact_id, depth)
        if artifact:
            return artifact, 200
        else:
            with Session() as session:
                artifact = (
                    session.query(Artifact)
                    .options(joinedload(Artifact.boilerplate_code_artifact))
                    .filter(Artifact.guid == artifact_id)
                    .first()
                )
                if artifact:
                    [artifact] = set_artifact_reference_depths([artifact], depth=depth)
                    artifact = marshal(artifact, artifact_model)
                    redis_cluster.set_artifact(artifact_id, depth, artifact)
            if artifact:
                return artifact, 200
        return {"message": f"Invalid artifact id provided: No artifact with id {artifact_id} exists."}, 400

    @ns.expect(artifact_model)
    def put(self, artifact_id):
        data = put_artifact_parser.parse_args()
        clear_cache = data.pop("clear_cache")
        artifact = None
        with Session(expire_on_commit=False) as session:
            artifact = session.query(Artifact).filter(Artifact.guid == artifact_id).first()
            if not artifact:
                artifact = Artifact(
                    content=data["content"],
                    type=data["type"],
                    title=data["title"],
                    creator=data["creator"],
                )
                session.add(artifact)
                try:
                    set_artifact_reference_fields(session, data, artifact)
                except KeyError as e:
                    session.rollback()
                    session.close()
                    return {"message": f"Artifact with guid {e} not found, no artifact with that guid exists."}, 400
            else:
                for field in ["content", "type", "title", "creator"]:
                    if data[field]:
                        setattr(artifact, field, data[field])
                delete_artifact_reference_fields(session, data, artifact)
                try:
                    set_artifact_reference_fields(session, data, artifact)
                except KeyError as e:
                    session.rollback()
                    session.close()
                    return {"message": f"Artifact with guid {e} not found, no artifact with that guid exists."}, 400
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                session.close()
                return {
                    "message": "Unable to update the artifact, there is a foreign key specified that doesn't exist."
                }, 400
            marshalled = marshal(artifact, artifact_model)
            # need to clear single artifact keys as well as all artifacts, potentially slow
            clear_cache_and_update_artifact(redis_cluster, marshalled, clear_cache)

        return artifact.guid, 200


@ns.route("/metadata/<artifact_guid>")
class ArtifactMetadata(Resource):
    def get(self, artifact_guid):
        with Session() as session:
            metadata = {}
            artifacts = (
                session.query(Artifact.id, Metadata.key, Metadata.value)
                .join(Metadata, Artifact.id == Metadata.artifact_id)
                .filter(Artifact.guid == artifact_guid)
            ).all()
            for artifact in artifacts:
                if artifact.key in metadata:
                    metadata[artifact.key].append(artifact.value)
                else:
                    metadata[artifact.key] = [artifact.value]
        return metadata, 200


deprecate_parser = ns.parser()
deprecate_parser.add_argument("artifacts", type=str, action="append", location="values")


@ns.route("/deprecate/artifacts")
class DeprecateArtifacts(Resource):
    @ns.expect(deprecate_parser)
    def post(self):
        args = deprecate_parser.parse_args()
        with Session() as session:
            db.set_deprecated(session, args["artifacts"], True)
            session.commit()
        clear_cache_for_deprecated(redis_cluster, args["artifacts"])
        return {"message": "The operation was successful."}, 200


@ns.route("/undeprecate/artifacts")
class UndeprecateArtifacts(Resource):
    @ns.expect(deprecate_parser)
    def post(self):
        args = deprecate_parser.parse_args()
        with Session() as session:
            db.set_deprecated(session, args["artifacts"], False)
            session.commit()
        clear_cache_for_deprecated(redis_cluster, args["artifacts"])
        return {"message": "The operation was successful."}, 200


user_artifact_parser = ns.parser()
user_artifact_parser.add_argument("content", type=str, required=True)
user_artifact_parser.add_argument("response_to", type=str, required=True)

get_user_artifact_parser = ns.parser()
get_user_artifact_parser.add_argument("response_to", type=str, location="values", required=False, default=None)


@ns.route("/user-artifacts")
class UserArtifacts(Resource):
    @ns.expect(get_user_artifact_parser)
    def get(self):
        response_to = get_user_artifact_parser.parse_args().get("response_to")
        user_artifacts = []
        with Session() as session:
            query = session.query(
                UserArtifact.guid.label("user_artifact_guid"),
                UserArtifact.content,
                Artifact.guid.label("artifact_guid"),
            ).join(Artifact, Artifact.id == UserArtifact.response_to)
            if response_to:
                query = query.filter(Artifact.guid == response_to)
            user_artifacts = query.all()
            marshalled_user_artifacts = marshal(user_artifacts, user_artifact_model)
        return marshalled_user_artifacts, 200

    @ns.expect(user_artifact_model)
    def post(self):
        data = user_artifact_parser.parse_args()
        user_artifact = UserArtifact(content=data["content"])
        with Session(expire_on_commit=False) as session:
            response_to = session.query(Artifact).filter(Artifact.guid == data["response_to"]).first()
            user_artifact.response_to = response_to.id
            session.add(user_artifact)
            session.commit()
        return user_artifact.guid, 200


@ns.route("/user-artifacts/<user_id>")
class SingleUserArtifacts(Resource):
    def get(self, user_id):
        user_artifacts = []
        with Session() as session:
            user_artifacts = (
                session.query(
                    UserArtifact.guid.label("user_artifact_guid"),
                    UserArtifact.content,
                    Artifact.guid.label("artifact_guid"),
                )
                .join(Artifact, Artifact.id == UserArtifact.response_to)
                .join(UserHistoryV2, UserHistoryV2.user_artifact_id == UserArtifact.id)
                .filter(UserHistoryV2.user == user_id)
                .all()
            )
            marshalled_user_artifacts = marshal(user_artifacts, user_artifact_model)
        return marshalled_user_artifacts, 200


user_history_parser = ns.parser()
user_history_parser.add_argument("user", type=str, required=True)
user_history_parser.add_argument("artifact_id", type=str)
user_history_parser.add_argument("user_artifact_id", type=str)
user_history_parser.add_argument("ml_data", type=dict)
user_history_parser.add_argument("session", type=str)

get_user_history_parser = ns.parser()
get_user_history_parser.add_argument("user", type=str, required=False, location="values", action="split")
get_user_history_parser.add_argument(
    "from_date", type=inputs.date_from_iso8601, location="values", help="Format: year-month-day e.g. 2023-01-24"
)
get_user_history_parser.add_argument(
    "to_date", type=inputs.date_from_iso8601, location="values", help="Format: year-month-day e.g. 2023-01-24"
)
get_user_history_parser.add_argument("session", type=str, location="values")
get_user_history_parser.add_argument("artifact", type=str, location="values", action="split")
get_user_history_parser.add_argument("populate_user_artifacts", type=inputs.boolean, location="values")
get_user_history_parser.add_argument("populate_artifacts", type=inputs.boolean, location="values")
get_user_history_parser.add_argument("format", type=str, choices=["json", "csv"], default="json", location="values")


@ns.route("/user-history")
class UserHistories(Resource):
    @ns.expect(get_user_history_parser)
    def get(self):
        search_query = get_user_history_parser.parse_args()

        with Session() as session:
            user_history = session.query(UserHistoryV2)
            if search_query.get("user"):
                user_history = user_history.filter(UserHistoryV2.user.in_(search_query["user"]))
            if search_query.get("from_date"):
                user_history = user_history.filter(cast(UserHistoryV2.timestamp, Date) >= search_query["from_date"])
            if search_query.get("to_date"):
                user_history = user_history.filter(cast(UserHistoryV2.timestamp, Date) <= search_query["to_date"])
            if search_query.get("session"):
                user_history = user_history.filter(UserHistoryV2.session == search_query["session"])

            user_history = user_history.options(joinedload(UserHistoryV2.artifact)).options(
                joinedload(UserHistoryV2.user_artifact)
            )
            if search_query.get("artifact"):
                _artifacts = (
                    session.query(Artifact)
                    .filter(Artifact.guid.in_(search_query["artifact"]))
                    .options(subqueryload(Artifact.artifacts))
                    .all()
                )
                artifact_ids = []
                while len(_artifacts) > 0:
                    artifact = _artifacts.pop()
                    artifact_ids.append(artifact.id)
                    _artifacts.extend([order_obj.child for order_obj in artifact.artifacts])
                user_history = user_history.filter(
                    or_(
                        UserHistoryV2.artifact.has(Artifact.id.in_(artifact_ids)),
                        UserHistoryV2.user_artifact.has(UserArtifact.response_to.in_(artifact_ids)),
                    )
                )
            user_history = user_history.order_by(UserHistoryV2.timestamp).all()
            marshalled_user_history = marshal(user_history, user_history_model)
            user_history = {u.id: u for u in user_history}
            for marshalled in marshalled_user_history:
                set_user_history_id(user_history, marshalled, is_v1=False)
            if search_query.get("populate_user_artifacts"):
                marshalled_user_history = populate_user_artifacts(session, marshalled_user_history, user_artifact_model)
            if search_query.get("populate_artifacts"):
                marshalled_user_history = populate_artifacts(session, marshalled_user_history)
        if search_query.get("format") == "csv":
            for history_obj in marshalled_user_history:
                if history_obj.get("ml_data", {}).get("output") not in [None, '']:
                    history_obj["ml_data"]["output"].pop("activity_summary", None)
            csv = json_to_csv_file(marshalled_user_history)
            return send_file(csv, download_name="user_history.csv", as_attachment=True)
        return marshalled_user_history, 200

    @ns.expect([input_user_history_model])
    @ns.doc(
        params={
            "user": "The guid of the user",
            "artifact_id": "The artifact received by the user, if any",
            "user_artifact_id": "The user artifact corresponding to something the user responded to, if any",
            "ml_data": {
                "description": "Data relating to some ml model that was called for the artifact or user artifact in"
                ' question. Of the form: {"version": <the version of the model>, "name": <the name of '
                'the model>, "input": <the input to the model>, "output": <the output generated by '
                "the model>}",
                "type": "object",
            },
            "session": "the session id that userhistory belongs to",
        }
    )
    
    def post(self):
        data = request.get_json()
        if isinstance(data, list) and len(data) == 0:
            return {"message": "Please provide a non-empty list."}, 400
        if not isinstance(data, list):
            data = [data]
        ids = []
        
        with Session(expire_on_commit=False) as session:
            coding_feedback_response_manager = CodingFeedbackManager(session)
            for d in data:                 
                artifact_id = d.pop("artifact_id", None)
                user_artifact_id = d.pop("user_artifact_id", None)
                coding_feedback_response_guid = d.pop("coding_feedback_response_id", None)
                user_history_obj = UserHistoryV2(**d)

                if coding_feedback_response_guid:
                    coding_feedback_response = coding_feedback_response_manager.get_coding_feedback_response_by_guid(
                        coding_feedback_response_guid
                    )
                    user_history_obj.coding_feedback_response_id = coding_feedback_response.id
                if artifact_id:
                    artifact = get_artifact_by_guid(session, artifact_id)
                    if artifact:
                        user_history_obj.artifact_id = artifact.id
                    else:
                       return {"message": "Artifact not found"}, 400 
                elif user_artifact_id:
                    user_artifact = session.query(UserArtifact).filter(UserArtifact.guid == user_artifact_id).first()
                    if user_artifact:
                        user_history_obj.user_artifact_id = user_artifact.id
                    else:
                        return {"message": "User Artifact not found"}, 400 
                session.add(user_history_obj)
                session.flush()
                ids.append(user_history_obj.id)
            session.commit()
        return ids if len(ids) > 1 else ids[0], 200


@ns.route("/user-history/<user_id>")
class SingleUserHistory(Resource):
    def get(self, user_id):
        user_history = []
        with Session() as session:
            user_history = (
                session.query(UserHistoryV2)
                .options(joinedload(UserHistoryV2.artifact))
                .options(joinedload(UserHistoryV2.user_artifact))
                .filter(UserHistoryV2.user == user_id)
                .all()
            )

            marshalled_user_history = marshal(user_history, user_history_model)
            user_history = {u.id: u for u in user_history}
            for marshalled in marshalled_user_history:
                set_user_history_id(user_history, marshalled, is_v1=False)

        return marshalled_user_history, 200


user_parser = ns.parser()
user_parser.add_argument("name", type=str, required=True)


topic_parser = ns.parser()
topic_parser.add_argument("name", type=str, location="json", required=True)


@ns.route("/topics")
class Topics(Resource):
    def get(self):
        with Session() as session:
            topics = session.query(Topic).all()
        return marshal(topics, topic_model)

    @ns.expect(topic_parser)
    def post(self):
        topic = topic_parser.parse_args()
        with Session(expire_on_commit=False) as session:
            topic = Topic(name=topic["name"])
            session.add(topic)
            session.commit()
        return topic.name, 200


test_case_parser = ns.parser()
test_case_parser.add_argument(
    "guid",
    type=str,
    location="values",
    help="If this is updating an existing test case, put the existing guid here.",
)
test_case_parser.add_argument(
    "s3_path",
    type=str,
    location="values",
    help="If this is a new test case, put the s3 path of where you want the file to be saved to here.",
)
test_case_parser.add_argument("topics", type=str, default=None, location="values", action="split")
test_case_parser.add_argument("creator", type=str, default=None, location="values")
test_case_parser.add_argument("title", type=str, default=None, location="values")
test_case_parser.add_argument("test_case_file", type=FileStorage, location="files", required=True)


@ns.route("/test-cases")
class TestCases(Resource):
    @ns.expect(test_case_parser)
    def put(self):
        data = test_case_parser.parse_args()
        artifact_id = data["guid"]
        file = data["test_case_file"]
        bucket = None
        prefix = None
        if data["s3_path"]:
            bucket, prefix = split_s3_prefix_path(data["s3_path"])
        with Session() as session:
            try:
                artifact_id = update_programming_exercise_test_case(session, artifact_id, file, data, bucket, prefix)
            except (NotAResource, NotFound) as e:
                return str(e), 400
        redis_cluster.clear_cache()
        return artifact_id, 200


resource_parser = ns.parser()
resource_parser.add_argument("guid", type=str, location="values")
resource_parser.add_argument("topics", type=str, default=None, location="values", action="split")
resource_parser.add_argument("creator", type=str, default=None, location="values")
resource_parser.add_argument("title", type=str, default=None, location="values")
resource_parser.add_argument("image", type=FileStorage, location="files", required=True)


@ns.route("/resources")
class Resources(Resource):
    @ns.expect(resource_parser)
    def put(self):
        data = resource_parser.parse_args()
        artifact_id = data["guid"]
        with Session() as session:
            if artifact_id:
                resource = session.query(Artifact).filter(Artifact.guid == artifact_id).first()
                if resource:
                    if resource.type != ArtifactType.RESOURCE.value:
                        session.close()
                        return f"Artifact {artifact_id} is not a resource.", 400
                    for field in ["title", "creator"]:
                        if data[field]:
                            setattr(resource, field, data[field])
                    set_artifact_reference_fields(session, data, resource)
                else:
                    session.close()
                    return f"No artifact with id {artifact_id} exists.", 400
            else:
                resource = Artifact(
                    type=ArtifactType.RESOURCE.value,
                    title=data["title"],
                    creator=data["creator"],
                )
                session.add(resource)
                set_artifact_reference_fields(session, data, resource)

            image = data["image"]
            try:
                upload_file_to_aws(
                    image,
                    image.filename,
                    config.image_s3.bucket,
                    config.image_s3.prefix,
                )
            except ClientError as e:
                session.rollback()
                session.close()
                return str(e), 500
            resource.content = image.filename
            artifact_id = resource.guid
            session.commit()
        return artifact_id, 200


prog_s3_file_parser = ns.parser()
prog_s3_file_parser.add_argument("folder", type=str, default=None, location="values")


@ns.route("/programming-s3-files")
class ProgrammingS3Files(Resource):
    @ns.expect(prog_s3_file_parser)
    def put(self):
        files = request.files.getlist("files")
        folder = prog_s3_file_parser.parse_args().get("folder")
        s3_mapping = {}
        if folder is None:
            folder = str(uuid.uuid4())
        folder = f"coding-exercise/{folder}"
        with Session(expire_on_commit=False) as session:
            for file in files:
                filename = file.filename
                s3_path = join_s3_path(folder, filename)
                artifact = Artifact(type=ArtifactType.RESOURCE.value, content=s3_path)
                session.add(artifact)
                session.commit()
                try:
                    if filename.startswith("test_case"):
                        file_with_record_propery = add_record_property_to_code(file)
                        file_data = file_with_record_propery.read()
                        file_with_record_propery.seek(0)
                        feedback_mappings = upload_feedback_mappings_from_test_file_to_db(file_data, artifact.guid)
                        file_with_record_property_and_feedback_guids = replace_all_feedback_strings_with_guids(
                            file_with_record_propery, feedback_mappings
                        ).read()
                        upload_file_to_aws(
                            file_with_record_property_and_feedback_guids, filename, config.s3.bucket, folder
                        )
                    else:
                        upload_file_to_aws(
                            file, filename, config.s3.bucket, folder
                        )
                    basename, extension = os.path.splitext(filename)
                    s3_mapping[basename] = artifact.guid
                except ClientError as e:
                    session.rollback()
                    session.close()
                    return {"message": str(e)}, 500
            session.commit()
        if s3_mapping:
            redis_cluster.clear_cache(redis_cluster.ALL_ARTIFACTS_PREFIX)
        return s3_mapping, 200


@ns.route("/programming-templates")
class ProgrammingTemplates(Resource):
    def get(self):
        prog_template_s3s = config.template_s3.programming_exercises
        test_case = read_s3_file(*split_s3_path(prog_template_s3s.test_case)).decode()
        boilerplate_code = read_s3_file(*split_s3_path(prog_template_s3s.boilerplate_code)).decode()
        answer = read_s3_file(*split_s3_path(prog_template_s3s.answer)).decode()
        return {
            "test_cases": [test_case],
            "boilerplate_code": boilerplate_code,
            "answers": [answer],
        }, 200


@ns.route("/programming-s3-files/<artifact_id>")
class ProgrammingArtifactS3(Resource):
    def get(self, artifact_id):
        with Session() as session:
            artifact = (
                session.query(Artifact)
                .filter(Artifact.guid == artifact_id)
                .options(
                    subqueryload(Artifact.artifacts),
                    subqueryload(Artifact.answers),
                    subqueryload(Artifact.test_cases),
                    subqueryload(Artifact.boilerplate_code_artifact),
                    subqueryload(Artifact.programming_resources),
                    subqueryload(Artifact.topics),
                )
                .first()
            )
            if not artifact or artifact.type != ArtifactType.PROGRAMMING_EXERCISE.value:
                session.close()
                return {"message": f"No programming exercise exists with guid {artifact_id}."}, 400
            response = retrieve_programming_exercise_s3_content(artifact_v2)
        return response, 200


prog_resource_parser = ns.parser()
prog_resource_parser.add_argument("folder", type=str, default=None, location="values", required=True)


@ns.route("/programming-resource-s3")
class ProgrammingResourcesS3(Resource):
    @ns.expect(prog_s3_file_parser)
    def put(self):
        files = request.files.getlist("programming_resource")
        folder = f"coding-exercise/{prog_s3_file_parser.parse_args()['folder']}"

        artifacts = []
        with Session(expire_on_commit=False) as session:
            for file in files:
                s3_path = join_s3_path(folder, file.filename)
                upload_file_to_aws(file, file.filename, config.s3.bucket, folder)
                artifact = Artifact(type=ArtifactType.RESOURCE.value, content=s3_path)
                session.add(artifact)
                session.commit()
                artifacts.append(artifact)
            artifacts = marshal(artifacts, artifact_model)
        if artifacts:
            redis_cluster.clear_cache(redis_cluster.ALL_ARTIFACTS_PREFIX)
        return artifacts, 200


programming_resource_parser = test_case_parser.copy()
programming_resource_parser.remove_argument("test_case_file")
programming_resource_parser.add_argument("resource_file", type=FileStorage, location="files", required=True)
programming_resource_parser.replace_argument(
    "guid",
    type=str,
    location="values",
    help="If this is updating an existing resource, put the existing guid here.",
)
programming_resource_parser.replace_argument(
    "s3_path",
    type=str,
    location="values",
    help="If this is a new resource, put the s3 path of where you want the file to be saved to here.",
)


@ns.route("/programming-resources")
class ProgrammingResources(Resource):
    @ns.expect(programming_resource_parser)
    def put(self):
        data = programming_resource_parser.parse_args()
        artifact_id = data["guid"]
        file = data["resource_file"]
        bucket = None
        prefix = None
        old_artifact = None
        if data["s3_path"]:
            bucket, prefix = split_s3_prefix_path(data["s3_path"])
        with Session() as session:
            try:
                artifact_id, artifact = update_programming_resource(session, artifact_id, file, data, bucket, prefix)
            except (NotAResource, NotFound) as e:
                return str(e), 400
            clear_cache_and_update_artifact(redis_cluster, marshal(artifact, artifact_model))
        return artifact_id, 200
